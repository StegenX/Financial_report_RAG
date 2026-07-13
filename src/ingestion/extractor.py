import logging
from pathlib import Path
import sys
import pymupdf
import pdfplumber

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)

import config

logging.basicConfig(
    filename=config.LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def detect_page_type(page_fitz, page_plumber):
    pass

def extract_text_page(page_fitz):
    pass

def extract_table_page(page_plumber):
    pass

def extract_image_page(page_fitz):
    pass

def extract_pdf(file_path, ticker, year):
    try:
        doc_pymupdf = pymupdf.open(file_path)
        doc_pdfplumber = pdfplumber.open(file_path)

        processed_pages = []

        for page_number in range(len(doc_pymupdf)):
            

    except Exception as e:
        logger.error(f"EXTRACT_PDF: Error: {e}")
        raise