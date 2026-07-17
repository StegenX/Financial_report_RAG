import logging
from pathlib import Path
import sys
import pymupdf
import pdfplumber
import re
import pytesseract
from PIL import Image
import json
from datetime import datetime

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)

import config

logging.basicConfig(
    filename=config.LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def detect_page_type(page_fitz, page_plumber) -> list[str]:
    try:
        logger.info("DETECT_PAGE_TYPE: Evaluating page content type...")

        types = []
        tables = page_plumber.find_tables()
        if tables:
            types.append("table")
        
        text = page_fitz.get_text().strip()
        if len(text) > 50:
            types.append("text")
        
        if not types:
            types.append("image")
        
        return types
    except Exception as e:
        logger.error(f"DETECT_PAGE_TYPE: Error: {e}")
        raise

def extract_text_page(page_fitz):
    try:
        logger.info(f"EXTRACT_TEXT_PAGE: Extracting text from page {page_fitz.number} in process ...")

        text: str = page_fitz.get_text("text")

        if not text:
            logger.warning(f"EXTRACT_TEXT_PAGE: {page_fitz.number} containes no text ...")
            return ""
        
        cleaned = text.replace("\f", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines())

        logger.info(f"EXTRACT_TEXT_PAGE: Extraction of text from page {page_fitz.number} done successfully ...")
        return cleaned.strip()
    except Exception as e:
        logger.error(f"EXTRACT_TEXT_PAGE: Error: {e}")
        try:
            return page_fitz.get_text("text").strip()
        except Exception:
            return ""
        

def extract_table_page(page_plumber):
    try:
        logger.info(f"EXTRACT_TABLE_PAGE: Extracting table from page {page_plumber.page_number} in process ...")

        tables = page_plumber.extract_tables()

        if not tables:
            logger.warning(f"EXTRACT_TABLE_PAGE: {page_plumber.page_number} contained no tables ...")
            return ""
        markdown_tables = []

        for table in tables:
            if not table or len(table) == 0:
                continue
            table_lines = []

            for row_idx, row in enumerate(table):
                cleaned_row = []
                for cell in row:
                    if cell is None:
                        cleaned_row.append("")
                    else:
                        cleaned_cell = str(cell).strip().replace("\n", " ")
                        cleaned_row.append(cleaned_cell)
                
                markdown_row = "| " + " | ".join(cleaned_row) + " |"
                table_lines.append(markdown_row)
                
                if row_idx == 0:
                    num_columns = len(cleaned_row)
                    separator = "| " + " | ".join(["---"] * num_columns) + " |"
                    table_lines.append(separator)
            
            markdown_tables.append("\n".join(table_lines))

        logger.info(f"EXTRACT_TABLE_PAGE: {page_plumber.page_number} Table extraction done successfully ...")

        return "\n\n".join(markdown_tables)

    except Exception as e:
        logger.error(f"EXTRACT_TABLE_PAGE: Error: {e}")
        return ""

def extract_image_page(page_fitz):
    try:
        logger.info(f"EXTRACT_IMAGE_PAGE: Extacting image from page {page_fitz.number} in process ...")

        pix = page_fitz.get_pixmap(dpi=300, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        raw_ocr_txt = pytesseract.image_to_string(img, config="--psm 3")

        if not raw_ocr_txt:
            return ""
        
        cleaned = raw_ocr_txt.strip()

        logger.info(f"EXTRACT_IMAGE_PAGE: {page_fitz.number} Text image extraction done successfully ...")

        return cleaned
    except Exception as e:
        logger.error(f"EXTRACT_IMAGE_PAGE: Error {e}")
        return ""

def extract_pdf(file_path, ticker, year):
    try:
        logger.info(f"EXTRACT_PDF: {file_path.rsplit('/')[-1]} File is in extraction process ...")
        output_path = Path(f"{config.PROCESSED_PATH}/{ticker}_{year}_processed.json")
        if output_path.exists():
            logger.info(f"EXTRACT_PDF: {ticker}_{year} already processed, skipping...")
            return
        
        doc_pymupdf = pymupdf.open(file_path)
        doc_pdfplumber = pdfplumber.open(file_path)

        processed_pages = []

        for page_number in range(len(doc_pymupdf)):
            page_fitz = doc_pymupdf[page_number]
            page_plumber = doc_pdfplumber.pages[page_number]

            page_type = detect_page_type(page_fitz, page_plumber)
            text_content = ""
            table_content = ""
            image_content = ""

            if "text" in page_type:
                text_content = extract_text_page(page_fitz)

            if "table" in page_type:
                table_content = extract_table_page(page_plumber)

            if "image" in page_type:
                image_content = extract_image_page(page_fitz)

            processed_pages.append({
                    "ticker": ticker,
                    "year": year,
                    "page_number": page_number + 1,
                    "content_types": page_type,
                    "text_content": text_content,
                    "table_content": table_content,
                    "image_content": image_content,
                })

        text_pages = sum(1 for p in processed_pages if "text" in p["content_types"])
        table_pages = sum(1 for p in processed_pages if "table" in p["content_types"])
        image_pages = sum(1 for p in processed_pages if "image" in p["content_types"])
        logger.info(f"EXTRACT_PDF: {ticker}_{year} — {text_pages} text, {table_pages} table, {image_pages} image pages")

        with open(f"{config.PROCESSED_PATH}/{ticker}_{year}_processed.json", 'w') as f:
            json.dump(processed_pages, f, indent=2)

        logger.info(f"EXTRACT_PDF: {file_path.rsplit('/')[-1]} File has been extracted successfuly ...")

        return True
    except Exception as e:
        logger.error(f"EXTRACT_PDF: Error: {e}")
        raise
    finally:
        if 'doc_pymupdf' in locals():
            doc_pymupdf.close()
        if 'doc_pdfplumber' in locals():
            doc_pdfplumber.close()


def extract_all(tickers, years):
    try:
        logger.info(f"EXTRACT_ALL: Extracting PDFs in process ...")
        succeded_extractions = []
        failed_extractions = []
        for ticker in tickers:
            for year in years:
                try:
                    file_path = f"{config.RAW_PATH}/{ticker.lower()}_{year}_10k.pdf"
                    extract_pdf(file_path, ticker.lower(), year)
                    succeded_extractions.append({"FILE_NAME": file_path.rsplit('/')[-1]})
                except Exception as e:
                    failed_extractions.append({"FILE_NAME": file_path.rsplit('/')[-1]})
            
        logger.info(f"EXTRACT_ALL: Extraction phase completed, {len(succeded_extractions)} success, {len(failed_extractions)} failed ...")

    except Exception as e:
        logger.error(f"EXTRACT_ALL: Error {e}")
        raise


if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Extract pipeline started at {start_time} ...")
        YEARS = config.YEARS
        TICKERS = config.TICKERS
        extract_all(TICKERS, YEARS)
        end_time = datetime.now()
        logger.info(f"Extract pipeline ended at {end_time}, {end_time - start_time} tim in total run  ...")
    except Exception as e:
        logger.error(f"Extract pipeline error: {e}")