import chromadb
import logging
from pathlib import Path
import sys
import time
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

logger = logging.getLogger(__name__)

def create_chroma_client():
    try:
        logger.info(f"CREATE_CHROMA_CLIENT: Creating presistent chromaDb client in process ...")

        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        logger.info(f"CREATE_CHROMA_CLIENT: Creating presistent chromaDb client done successfully ...")

        return client
    except Exception as e:
        logger.error(f"CREATE_CHROMA_CLIENT: Error: {e}")
        raise


def get_or_create_collection(client, collection_name):
    try:
        logger.info(f"GET_OR_CREATE_COLLECTION: Creating or retreiving collection {collection_name} in process ...")

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"GET_OR_CREATE_COLLECTION: Successfully accessed collection '{collection_name}'.")
        
        return collection
    except Exception as e:
        logger.error(f"GET_OR_CREATE_COLLECTION: Error: {e}")
        raise

def index_document(collection: chromadb.Collection, embedded_chunks):
    try:
        logger.info(f"INDEX_DOCUMENT: Indexing and storing embedded chunks in process ...")

        batch_size = 100
        indexed = 0
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i: i + batch_size]

            metadatas=[{
                "ticker": chunk["metadata"]["ticker"],
                "year": chunk["metadata"]["year"],
                "page_number": chunk["metadata"]["page_number"],
                "chunk_type": chunk["chunk_type"],
                "source": chunk["source"]
            } for chunk in batch]

            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["content"] for chunk in batch],
                embeddings=[chunk["embedding"] for chunk in batch],
                metadatas=metadatas
            )

            indexed += len(batch)

        logger.info(f"INDEX_DOCUMENT: Indexing and storing embedded chunks done successfully, {indexed} chunks has been indexed and stored ...")
        return indexed
    except Exception as e:
        logger.error(f"INDEX_DOCUMENT: Error: {e}")
        raise

def index_all(tickers, years, client, collection):
    try:
        logger.info(f"INDEX_ALL: Indexing and storing phase in process ...")

        indexed = 0
        success = []
        fail = []

        for ticker in tickers:
            for year in years:
                embeddings_path = Path(f"{config.EMBEDDINGS_PATH}/{ticker.lower()}_{year}_embeddings.json")

                if not embeddings_path.exists():
                    logger.info(f"EMBED_DOCUMENT: No embedding found for {ticker.lower()}_{year}, Skipping ...")
                    continue
                with open(embeddings_path, 'r') as f:
                    embedded_chunks = json.load(f)
                try:
                    indexed += index_document(collection, embedded_chunks)
                    success.append({"Document": f"{ticker.lower()}_{year}_10k.pdf"})
                except Exception as e:
                    fail.append({"Document": f"{ticker.lower()}_{year}_10k.pdf"})
                    logger.error(f"EMBED_ALL: Indexing and storing for {ticker.lower()}_{year}_10k.pdf failed, Error : {e}")

        logger.info(f"INDEX_ALL: Indexing and storing phase completed, {len(success)} successded, {len(fail)} failed, {indexed} index created and stored...")

    except Exception as e:
        logger.error(f"INDEX_ALL: Error: {e}")
        raise

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Indexing pipeline started at {start_time} ...")
        YEARS = config.YEARS
        TICKERS = config.TICKERS
        index_all(TICKERS, YEARS)
        end_time = datetime.now()
        logger.info(f"Indexing pipeline ended at {end_time}, {end_time - start_time} time in total run  ...")
    except Exception as e:
        logger.error(f"Indexing pipeline failed: Error {e}")
