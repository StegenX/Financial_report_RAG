import logging
from pathlib import Path
from sentence_transformers import CrossEncoder
import sys
import time
from datetime import datetime

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)

import config
from src.indexing.embedder import load_embedding_model
from src.indexing.vector_store import create_chroma_client, get_or_create_collection
# from retriever import create_retriever, retrieve, format_context

LOGS_PATH=config.LOGS_PATH

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        logger.info(f"LOAD_RERANKER: Loading reranker model in process ...")

        start_time = time.time()

        model = CrossEncoder(model_name)

        logger.info(f"LOAD_RERANKER: Loading reranker loaded successfully in {time.time() - start_time} seconds ...")
        return model
    except Exception as e:
        logger.error(f"LOAD_RERANKER: Error: {e}")
        raise

def rerank(query, candidates, reranker: CrossEncoder, top_k=5):
    try:
        if not query or not query.strip():
            logger.warning("RETRIEVE: Received an empty query string. Returning empty results.")
            return []
        if not candidates:
            logger.warning("RETRIEVE: Received an empty candidates chunks. Returning empty results.")

        logger.info(f"RERANK: Reranking {len(candidates)} candidates for query. Target top_k={top_k}")

        pairs = [[query, chunk.get("content", "")] for chunk in candidates]

        scores = reranker.predict(pairs)

        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = score

        reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        top_score = reranked_candidates[0]["rerank_score"]
        bottom_score = reranked_candidates[-1]["rerank_score"]
        logger.info(f"RERANK: Optimization complete. Max score: {top_score:.4f} | Min score: {bottom_score:.4f}")

        return reranked_candidates[:top_k]
    except Exception as e:
        logger.error(f"RERANK: Failed to execute cross-encoder optimization: {e}")
        raise
    
# if __name__ == "__main__":
#     try:
#         logger.info(f"Reranking pipeline in process ...")
#         client = create_chroma_client()
#         collection = get_or_create_collection(client, config.COLLECTION_NAME)
#         model = load_embedding_model()
        
#         retriever_bundle = create_retriever(collection, model)
        
#         query = "What was Apple's total revenue in 2023?"
#         search_filter = {"ticker": "aapl", "year": 2023}
        
#         chunks = retrieve(retriever_bundle, query, filters=search_filter)
#         reranker_model = load_reranker()
#         top_relevent = rerank(query, chunks, reranker_model)
#         context = format_context(top_relevent)
#     except Exception as e:
#         logger.error(f"Reranking pipeline failed: {e}")