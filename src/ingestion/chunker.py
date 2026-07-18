import json
import logging
from pathlib import Path
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def chunk_text(text, metadata):
    try:
        page_num = metadata.get("page_number")
        ticker = metadata.get("ticker", "UNKNOWN")
        year = metadata.get("year", "UNKNOWN")
        source = f"{ticker}_{year}_10k.pdf"
        logger.info(f"CHUNK_TEXT: Chunking text for page {page_num} in process ...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        fragements = text_splitter.split_text(text)
        chunks = []

        if fragements:
            for index, fragement in enumerate(fragements):
                chunks.append({
                    "chunk_id": f"{ticker}_{year}_p{page_num}_text_{index}",
                    "content": fragement,
                    "chunk_type": "text",
                    "metadata": metadata,
                    "source": source
                })

        logger.info(f"CHUNK_TEXT: Page {page_num} text chunking done successfully ...")

        return chunks
    except Exception as e:
        logger.error(f"CHUNK_TEXT: Error: {e}")
        return []


def chunk_table(table_content, metadata):
    try:
        page_num = metadata.get("page_number")
        ticker = metadata.get("ticker", "UNKNOWN")
        year = metadata.get("year", "UNKNOWN")
        source = f"{ticker}_{year}_10k.pdf"
        logger.info(f"CHUNK_TABLE: Chunking table for page {page_num} in process ...")

        tables = table_content.split("\n\n")
        chunks = []

        for index, table in enumerate(tables):
            if table.strip():
                chunks.append({
                    "chunk_id": f"{ticker}_{year}_p{page_num}_table_{index}",
                    "content": table,
                    "chunk_type": "table",
                    "metadata": metadata,
                    "source": source
                })

        logger.info(f"CHUNK_TABLE: Page {page_num} table chunking done successfully ...")

        return  chunks

    except Exception as e:
        logger.error(f"CHUNK_TABLE: Error: {e}")
        return []

def chunk_image(image_content, metadata):
    try:
        page_num = metadata.get("page_number")
        ticker = metadata.get("ticker", "UNKNOWN")
        year = metadata.get("year", "UNKNOWN")
        source = f"{ticker}_{year}_10k.pdf"
        logger.info(f"CHUNK_IMAGE: Chunking image text for page {page_num} in process ...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        fragements = text_splitter.split_text(image_content)
        chunks = []

        if fragements:
            for index, fragement in enumerate(fragements):
                chunks.append({
                    "chunk_id": f"{ticker}_{year}_p{page_num}_image_{index}",
                    "content": fragement,
                    "chunk_type": "image",
                    "metadata": metadata,
                    "source": source
                })

        logger.info(f"CHUNK_IMAGE: Page {page_num} image chunking done successfully ...")

        return chunks
    except Exception as e:
        logger.error(f"CHUNK_IMAGE: Error: {e}")
        return []

def chunk_page(page):
    try:
        page_num = page.get("page_number", 0)
        logger.info(f"CHUNK_PAGE: Chunking page number {page_num} in process ...")

        ticker = page.get("ticker", "UNKNOWN")
        year = page.get("year", "UNKNOWN")
        base_metadata = {
            "ticker": ticker,
            "year": year,
            "page_number": page_num 
        }

        page_chunks = []

        page_types = page.get("content_types")

        if "table" in page_types:
            table_content = page.get("table_content", "").strip()
            if table_content:
                table_chunks = chunk_table(table_content, base_metadata)
                if table_chunks:
                    page_chunks.extend(table_chunks)
        if "text" in page_types:
            text_content = page.get("text_content", "").strip()
            if text_content:
                text_fragements = chunk_text(text_content, base_metadata)
                if text_fragements:
                    page_chunks.extend(text_fragements)
            
        if "image" in page_types:
            image_content = page.get("image_content", "").strip()
            if image_content:
                image_text_fragements = chunk_image(image_content, base_metadata)
                if image_text_fragements:
                    page_chunks.extend(image_text_fragements)

        logger.info(f"CHUNK_PAGE: Page {page_num} chunking done successfully, {len(page_chunks)} chunk retrived ...")
        
        return page_chunks
    except Exception as e:
        logger.error(f"CHUNK_PAGE: Error: {e}")
        raise

def chunk_document(ticker, year):
    try:
        logger.info(f"CHUNK_DOCUMENT: Chucnking {ticker}_{year} data in process ...")

        processed_json_path = Path(f"{config.PROCESSED_PATH}/{ticker}_{year}_processed.json")
        chunks_path = Path(f"{config.CHUNKS_PATH}/{ticker}_{year}_chunks.json")

        if chunks_path.exists():
            logger.info(f"CHUNK_DOCUMENT: {ticker}_{year} chunks already exists, Loading from disk ...")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        if not processed_json_path.exists():
            logger.error(f"CHUNK_DOCUMENT: File not found {processed_json_path} ...")
            return []
        
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            processed_pages = json.load(f)

        all_chunks = []
        stats = {"text": 0, "table": 0, "image": 0}

        for page in processed_pages:
            page_chunk = chunk_page(page)

            all_chunks.extend(page_chunk)

            for chunk in page_chunk:
                c_type = chunk.get("chunk_type", "unknown")
                if c_type in stats:
                    stats[c_type] += 1

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2)

        logger.info(f"CHUNK_DOCUMENT: Successfully chunked {ticker.upper()} {year}.")
        logger.info(f"CHUNK_DOCUMENT: [METRICS] Total Chunks: {len(all_chunks)} | Text: {stats['text']} | Table: {stats['table']} | Image: {stats['image']}")
        
        return all_chunks
    except Exception as e:
        logger.error(f"CHUNK_DOCUMENT: Error {e}")
        raise

        

def chunk_all(tickers, years):
    try:
        logger.info(f"CHUNK_ALL: Chunking documents in process ...")
        success = []
        failure = []

        for ticker in tickers:
            for year in years:
                try:
                    chunk_document(ticker.lower(), year)
                    success.append({"Document": f"{ticker}_{year}"})
                except Exception as e:
                    failure.append({"Document": f"{ticker}_{year}"})
                    logger.error(f"CHUNK_ALL: Error: {e}")

        
        logger.info(f"CHUNK_ALL: Chunking documents completed, {len(success)} succeseded, {len(failure)} failed ...")
    except Exception as e:
        logger.error(f"CHUNK_ALL: Error: {e}")

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Chunk pipeline started at {start_time} ...")
        YEARS = config.YEARS
        TICKERS = config.TICKERS
        chunk_all(TICKERS, YEARS)
        end_time = datetime.now()
        logger.info(f"Chunk pipeline ended at {end_time}, {end_time - start_time} time in total run  ...")
    except Exception as e:
        logger.error(f"Chunk pipeline error: {e}")