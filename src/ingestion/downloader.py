import logging
import sys
from pathlib import Path
import httpx
import asyncio
from weasyprint import HTML
import os
import json
from datetime import datetime


parent_dir = str(Path(__file__).resolve().parent.parent.parent)

sys.path.append(parent_dir)

import config

LOGS_PATH=config.LOGS_PATH

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger  = logging.getLogger(__name__)

header = {
    "User-Agent": f"{config.USER} {config.EMAIL}"
}

def init_httpx_client() -> httpx.AsyncClient:
        try:
            logger.info(f"INIT_HTTPX_CLIENT: Initializing httpx connection ...")
            client = httpx.AsyncClient()
            logger.info(f"INIT_HTTPX_CLIENT: Initializing httpx connection succeded ...")
            return client
        except Exception as e:
            logger.error(f"INIT_HTTPX_CLIENT: Error : {e}")
            raise


def prepare_path(ticker: str, year: int) -> Path:
    try:
        logger.info(f"PREPARE_PATH: Resolving directory path ...")

        base_dir = Path(parent_dir) / "data" / "raw"
        base_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{ticker.lower()}_{year}_10k.pdf"

        file_path = base_dir / file_name

        logger.info(f"PREPARE_PATH: File path resolved successfully: {file_path}")

        return file_path
    except Exception as e:
        logger.error(f"PREPARE_PATH: Error: {e}")
        raise


async def get_CIK(ticker: str, client: httpx.AsyncClient):
    try:
        logger.info(f"GET_CIK: Extracting {ticker} CIK ...")

        tickers_url = 'https://www.sec.gov/files/company_tickers.json'
        response = await client.get(tickers_url, headers=header)

        if (response.status_code == 200):
            tickers = response.json()
            for company in tickers.values():
                if company["ticker"] == ticker:
                    cik = str(company['cik_str'])
                    cik = cik.zfill(10)

                    logger.info(f"GET_CIK: {ticker} CIK: {cik} has been retrived successfully ...")
                    return cik
                    
        logger.warning(f"GET_CIK: {ticker} CIK no found ...")           
        raise ValueError(f"{ticker} CIK not found")

    except Exception as e:
        logger.error(f"GET_CIK: Error: {e}")
        raise



async def get_filing_url(ticker: str, year: int, client: httpx.AsyncClient, ticker_cik: str):
    try:
        logger.info(f"GET_FILING_URL: Extracting PDF filing url for {ticker} from {year} ...")

        url = f'https://data.sec.gov/submissions/CIK{ticker_cik}.json'

        response = await client.get(url, headers=header)

        if(response.status_code == 200):
            data = response.json()
            filings = data['filings']['recent']

            form = filings["form"]
            filing_date = filings["filingDate"]
            accession_number = filings["accessionNumber"]
            primary_documents = filings["primaryDocument"]

            for index in range(len(form)):
                is_10k = form[index] == '10-K'
                match_year = str(filing_date[index]).startswith(f"{year}")

                if is_10k and match_year:
                    raw_accession = accession_number[index]
                    primary_doc = primary_documents[index]

                    clean_accession = raw_accession.replace("-", "")
                    unpadded_cik = ticker_cik.lstrip("0")

                    final_url = f"https://www.sec.gov/Archives/edgar/data/{unpadded_cik}/{clean_accession}/{primary_doc}"
                    logger.info(f"GET_FILING_URL: PDF Filing url for {ticker} from {year} has been retrived successfully ...")

                    return final_url

        logger.warning(f"GET_FILING_URL: No documents for {ticker} in year {year} ...")
        raise ValueError(f"No 10-K filing found for {ticker} in {year}")

    except Exception as e:
        logger.error(f"GET_FILING_URL: Error: {e}")
        raise


async def download_filing(ticker: str, year: int, client: httpx.AsyncClient, filing_url: str, file_path: Path):
    try:
        logger.info(f"DOWNLOAD_FILING: Downloading PDF filing for {ticker} from {year} ...")

        raw_bytes = bytearray()

        async with client.stream("GET", filing_url, headers=header) as response:
            if response.status_code == 200:
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    raw_bytes.extend(chunk)
        
        if filing_url.lower().endswith('.pdf'):
            logger.info(f"{ticker}_{year}_10k File is already a native PDF. Saving binary payload directly...")
            with open(file_path, 'wb') as f:
                f.write(raw_bytes)
        else:
            logger.info(f"{ticker}_{year}_10k File is HTML. Compiling to PDF via WeasyPrint...")
            html_content = raw_bytes.decode(encoding='utf-8', errors='ignore')
            HTML(string=html_content, base_url=filing_url).write_pdf(file_path)

        file_size = round(file_path.stat().st_size / (1024 * 1024), 2)
        
        logger.info(f"DOWNLOAD_FILING: {ticker} from {year} has been downloaded successfully, {file_size}mb added to disk...")
        return file_size
        
    except Exception as e:
        logger.error(f"DOWNLOAD_FILING: Error: {e}")
        raise

async def download_all(tickers: list[str], years: list[int]):
    try:
        logger.info(f"Ingestion pipeline start Downloading operation for {tickers} financial report ...")
        client = init_httpx_client()

        successfull_paths = []
        failed_paths = []
        CIKs = {}


        start_year = min(years)
        end_year = max(years) + 1
        total_size = 0

        for ticker in tickers:
            CIKs[ticker] = await get_CIK(ticker, client)


        for year in range(start_year ,end_year):
            for ticker, cik in CIKs.items():
                logger.info(f"INGESTION_PIPLINE: Start downloading {ticker}_{year}_10k.pdf ...")
                file_path = prepare_path(ticker, year)
                if os.path.exists(file_path):
                    logger.info(f"INGESTION_PIPLINE: {file_path} File already exists, Skipping ...")
                    continue
                try:
                    filing_url =  await get_filing_url(ticker.upper(), year, client, cik)
                    total_size += await download_filing(ticker, year, client, filing_url, file_path)
                    if os.path.exists(file_path):
                        successfull_paths.append(f"{ticker}_{year}_10k.pdf")
                    else:
                        raise FileNotFoundError(f"{file_path}")

                    logger.info(f"INGESTION_PIPLINE: {ticker}_{year}_10k.pdf Downloaded successfully ...")
                    await asyncio.sleep(1)
                except Exception as e:
                    failed_paths.append(f"{ticker}_{year}_10k.pdf")
                    logger.error(f"INGESTION_PIPLINE: {ticker}_{year}_10k.pdf Download failed, Check logs for more details ...")


        with open(f"{config.FAILED_PATHS}/failed_paths.json", 'w') as f:
            json.dump(failed_paths, f, indent=2)
        
        with open(f"{config.SUCCESSFULL_PATHS}/successfull_paths.json", 'w') as f:
            json.dump(successfull_paths, f, indent=2)

        logger.info(f"INGESTION_PIPLINE: Completed , {len(failed_paths)} failed, {len(successfull_paths)} succeded, {total_size} MB total written to disk ...")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        await client.aclose()
        logger.info("Closing htppx connection ...")
        logger.info("connection closed successfully ...")

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Ingestion pipeline started at {start_time} ...")
        YEARS = config.YEARS
        TICKERS = config.TICKERS
        asyncio.run(download_all(TICKERS, YEARS))
        end_time = datetime.now()
        logger.info(f"Ingestion pipeline ended at {end_time}, {end_time - start_time} time in total run  ...")
    except Exception as e:
        logger.error(f"Ingestion pipeline error: {e}")
