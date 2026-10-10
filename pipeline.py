from src.ingestion.downloader import download_all
from src.ingestion.extractor import extract_all
from src.ingestion.chunker import chunk_all
from src.indexing.embedder import embed_all, load_embedding_model
from src.indexing.vector_store import index_all, create_chroma_client, get_or_create_collection
from src.retrieval.reranker import load_reranker, rerank
from src.retrieval.retriever import create_retriever, bm25_retrieve, retrieve, hybrid_retrieve ,format_context
from src.generation.generator import answer, load_nvidia_client, generate

import logging
import config
from datetime import datetime
import asyncio

LOGS_PATH=config.LOGS_PATH

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def run_pipeline(tickers, years):
    try:
        logger.info(f"RAG Pipeline started ...")

        start_time = datetime.now()
        logger.info(f"=== PIPELINE START: {start_time} ===")
        logger.info(f"Processing Target: Tickers={tickers} | Years={years}")

        logger.info("--- PHASE 1: DOWNLOAD STARTED ---")
        asyncio.run(download_all(tickers, years))
        logger.info("--- PHASE 1: DOWNLOAD COMPLETED ---")

        logger.info("--- PHASE 2: EXTRACT STARTED ---")
        extract_all(tickers, years)
        logger.info("--- PHASE 2: EXTRACT COMPLETED ---")

        logger.info("--- PHASE 3: CHUNK STARTED ---")
        chunk_all(tickers, years)
        logger.info("--- PHASE 3: CHUNK COMPLETED ---")

        logger.info("--- PHASE 4: EMBED STARTED ---")
        embed_all(tickers, years)
        logger.info("--- PHASE 4: EMBED COMPLETED ---")

        logger.info("--- PHASE 5: INDEX STARTED ---")
        index_all(tickers, years)
        logger.info("--- PHASE 5: INDEX COMPLETED ---")

        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"=== PIPELINE SUCCESS: Completed in {duration} ===")
        print(f"\n✅ Pipeline completed successfully in {duration}!")


    except Exception as e:
        logger.error(f"RAG Pipeline failed: Error: {e}")
        raise

if __name__ == "__main__":
    try:
        tickers = config.TICKERS
        years = config.YEARS
        run_pipeline(tickers, years)
    except Exception as e:
        logger.error(f"Error: {e}")
        
