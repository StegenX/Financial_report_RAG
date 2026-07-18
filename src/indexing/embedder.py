from sentence_transformers import SentenceTransformer
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

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    try:
        logger.info(f"LOAD_MODEL: Loading {model_name} model in process ...")

        start_time = time.time()

        model = SentenceTransformer(model_name)
        end_time = time.time()

        load_time = round((end_time - start_time), 2)
        logger.info(f"LOAD_MODEL: {model_name} is loaded successfully in {load_time} seconds,  ...")

        return model
    except Exception as e:
        logger.error(f"LOADING_MODEL: Error: {e}")
        raise

def embed_chunk(chunk: dict, model: SentenceTransformer):
    try:
        logger.debug(f"EMBED_CHUNK: Embedding chunk in process ...")

        content = chunk.get("content")
        if not content:
            logger.warning(f"EMBED_CHUNK: Empty content found for chunk ID '{chunk.get('chunk_id', 'UNKNOWN')}'. Skipping embedding ...")
            return None

        embedding = model.encode(content)
        embedding_list = embedding.tolist()

        logger.debug(f"EMBED_CHUNK: Chunk of ID '{chunk.get('chunk_id', 'UNKNOWN')}' embedding done successfully ...")
        return embedding_list
    except Exception as e:
        logger.error(f"EMBED_CHUNK: Error: {e}")
        return None
    
def embed_document(ticker, year, model: SentenceTransformer):
    try:
        logger.info(f"EMBED_DOCUMENT: Embedding for document {ticker}_{year}_10k.pdf in process ...")

        embeddings_path = Path(f"{config.EMBEDDINGS_PATH}/{ticker}_{year}_embeddings.json")

        if embeddings_path.exists():
            logger.info(f"EMBED_DOCUMENT: {ticker}_{year} already embedded, loading from disk...")
            with open(embeddings_path, 'r') as f:
                return json.load(f)

        with open(f"{config.CHUNKS_PATH}/{ticker}_{year}_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        embed_count = 0
        embd = []
        start_time = time.time()
        for chunk in chunks:
            embedding = embed_chunk(chunk, model)


            if not embedding:
                logger.warning(f"EMBED_DOCUMENT: No embedding found for chunk ID '{chunk.get('chunk_id')}' ...")
                continue
            
            chunk['embedding'] = embedding
            embed_count += 1
            embd.append(chunk)
        end_time = time.time()
        load_time = round(end_time - start_time, 2)

        with open(embeddings_path, 'w') as f:
                json.dump(embd, f, indent=2)

        logger.info(f"EMBED_DOCUMENT: Embedding for document {ticker}_{year}_10k.pdf done sucessfully in {load_time} seconds, {embed_count} embedding generated ...")

        return embd
    except Exception as e:
        logger.error(f"EMBED_DOCUMENT: Error: {e}")
        raise

def embed_all(tickers, years, model: SentenceTransformer):
    try:
        logger.info(f"EMBED_ALL: Embedding pipeline is in process ...")
        

        embeddings = []
        success = []
        fail = []

        for ticker in tickers:
            for year in years:
                chunk_path = Path(f"{config.CHUNKS_PATH}/{ticker.lower()}_{year}_chunks.json")
                if not chunk_path.exists():
                    logger.warning(f"EMBED_ALL: No chunks found for {ticker.lower()}_{year}_chunks.json, Skipping ...")
                    continue
                try:
                    doc_embeddding = embed_document(ticker.lower(), year, model)
                    success.append({"Document": f"{ticker.lower()}_{year}_10k.pdf"})
                    embeddings.extend(doc_embeddding)
                except Exception as e:
                    fail.append({"Document": f"{ticker.lower()}_{year}_10k.pdf"})
                    logger.error(f"EMBED_ALL: Embedding for {ticker.lower()}_{year}_10k.pdf failed, Error : {e}")


        
        logger.info(f"EMBED_ALL: Embedding pipeline completed successfully, {len(embeddings)} embeddings in total, {len(success)} successeded, {len(fail)} failed ...")
        return embeddings
    except Exception as e:
        logger.error(f"EMBED_ALL: Error: {e}")
        raise

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Embedding pipeline started at {start_time} ...")
        YEARS = config.YEARS
        TICKERS = config.TICKERS
        model = load_embedding_model()
        embeddings = embed_all(TICKERS, YEARS, model)
        end_time = datetime.now()
        logger.info(f"Embedding pipeline ended at {end_time}, {end_time - start_time} time in total run  ...")
    except Exception as e:
        logger.error(f"Embedding pipeline error: {e}")