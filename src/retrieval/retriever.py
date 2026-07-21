import chromadb
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
from rank_bm25 import BM25Okapi
import json

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)


import config
from src.indexing.embedder import load_embedding_model
from src.indexing.vector_store import create_chroma_client, get_or_create_collection

LOGS_PATH=config.LOGS_PATH

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def create_retriever(collection, model):
    try:
        logger.info(f"CREATE_RETRIEVER: Bundling collection '{collection.name}' with embedding model.")
        
        retriever = {
            "collection": collection,
            "model": model
        }
        
        return retriever
    except Exception as e:
        logger.error(f"CREATE_RETRIEVER: Error: {e}")
        raise

def bm25_retrieve(query: str, filters: dict, n_results: int = 20) -> list[dict]:
    try:
        if not query or not query.strip():
            logger.warning("BM25_RETRIEVE: Received an empty query string. Returning empty results.")
            return []

        ticker = str(filters.get("ticker", "")).lower()
        year = str(filters.get("year", ""))

        if not ticker or not year:
            logger.error("BM25_RETRIEVE: Both 'ticker' and 'year' must be supplied in filters.")
            return []

        target_filename = f"{ticker}_{year}_chunks.json"
        target_path = Path(config.CHUNKS_PATH) / target_filename

        if not target_path.exists():
            logger.error(f"BM25_RETRIEVE: Target chunk file not found at path: {target_path}")
            return []

        logger.info(f"BM25_RETRIEVE: Loading targeted corpus file: {target_filename}")
        
        with open(target_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            logger.warning(f"BM25_RETRIEVE: File {target_filename} contains no chunks.")
            return []

        tokenized_corpus = [chunk.get("content", "").lower().split() for chunk in chunks]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()

        top_chunks = bm25.get_top_n(tokenized_query, chunks, n=n_results)

        scores = bm25.get_scores(tokenized_query)
        for i, chunk in enumerate(chunks):
            chunk["bm25_score"] = float(scores[i])
        top_chunks = sorted(chunks, key=lambda x: x["bm25_score"], reverse=True)[:n_results]

        logger.info(f"BM25_RETRIEVE: Retrieved top {len(top_chunks)} keyword matches from {target_filename}.")
        return top_chunks

    except Exception as e:
        logger.error(f"BM25_RETRIEVE: Failed to execute targeted lexical search: {e}")
        raise

def retrieve(retriever: dict, query: str, n_results: int = 20, filters: dict = None):
    try:
        model: SentenceTransformer = retriever["model"]
        collection: chromadb.Collection = retriever["collection"]

        if not query or not query.strip():
            logger.warning("RETRIEVE: Received an empty query string. Returning empty results.")
            return []

        logger.info(f"RETRIEVE: Processing query: '{query}' | Requesting n_results={n_results}")
        start_time = time.time()
        
        embeddings = model.encode(query).tolist()

        where_clause = None
        if filters:
            formatted_filters = []
            for key, value in filters.items():

                if key == "ticker" and isinstance(value, str):
                    val = value.lower()
                elif key == "year":
                    val = int(value)
                else:
                    val = value
                formatted_filters.append({key: {"$eq": val}})
            
            if len(formatted_filters) == 1:
                where_clause = formatted_filters[0]
            elif len(formatted_filters) > 1:
                where_clause = {"$and": formatted_filters}
                
            logger.debug(f"RETRIEVE: Translated filters to ChromaDB schema: {where_clause}")

        results = collection.query(
            query_embeddings=[embeddings],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        if results and results.get("documents"):
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]
            distance = results["distances"][0]
            for i in range(0, len(documents)):
                retrieved_chunks.append({
                    "chunk_id": ids[i],
                    "content": documents[i],
                    "distance": round(distance[i], 2),
                    "metadatas": metadatas[i]
                })

        duration = round(time.time() - start_time, 0)

        logger.info(f"RETRIEVE: Search completed successfully in {duration} seconds. Found {len(retrieved_chunks)} matches.")
        return retrieved_chunks

    except Exception as e:
        logger.error(f"RETRIEVER: Error: {e}")
        raise

def hybrid_retrieve(retriever: dict, query: str, n_results: int = 20, filters: dict = None) -> list[dict]:
    try:
        if not query or not query.strip():
            logger.warning("HYBRID_RETRIEVE: Received empty query. Returning empty results.")
            return []

        logger.info(f"HYBRID_RETRIEVE: Starting hybrid search for query: '{query}'")

        fetch_k = max(n_results * 3, 60) 
        
        semantic_results = retrieve(retriever, query, n_results=fetch_k, filters=filters)
        bm25_results = bm25_retrieve(query, filters=filters, n_results=fetch_k)
        
        fused_results = {}
        k_constant = 60
        
        for rank, chunk in enumerate(semantic_results, start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in fused_results:
                fused_results[chunk_id] = chunk
                fused_results[chunk_id]["rrf_score"] = 0.0
                
            fused_results[chunk_id]["rrf_score"] += 1.0 / (rank + k_constant)
            
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
                
            if chunk_id not in fused_results:
                fused_results[chunk_id] = chunk
                fused_results[chunk_id]["rrf_score"] = 0.0
                
            fused_results[chunk_id]["rrf_score"] += 1.0 / (rank + k_constant)

        sorted_fused_chunks = sorted(
            fused_results.values(), 
            key=lambda x: x["rrf_score"], 
            reverse=True
        )
        
        final_results = sorted_fused_chunks[:n_results]
        
        logger.info(f"HYBRID_RETRIEVE: Fused {len(semantic_results)} semantic and {len(bm25_results)} lexical chunks into {len(final_results)} final results.")
        return final_results
        
    except Exception as e:
        logger.error(f"HYBRID_RETRIEVE: Failed to execute RRF hybrid search: {e}")
        return []
    
def format_context(results):
    try:
        if not results:
            logger.warning("FORMAT_CONTEXT: Received empty results list. Returning empty context string.")
            return ""
            
        logger.info(f"FORMAT_CONTEXT: Formatting {len(results)} chunks into prompt context...")
        
        context_blocks = []
        
        for chunk in results:
            content = chunk.get("content", "").strip()
            if not content:
                continue
                
            metadata = chunk.get("metadatas", {})
            
            ticker = metadata.get("ticker", "UNKNOWN").lower()
            year = metadata.get("year", "UNKNOWN")
            page_num = metadata.get("page_number", "UNKNOWN")
            
            chunk_type = metadata.get("chunk_type")
            header = f"[Source: {ticker}_{year}_10k.pdf | Page: {page_num} | Type: {chunk_type}]"
            
            block = f"{header}\n{content}"
            context_blocks.append(block)
            
        formatted_context = "\n\n".join(context_blocks)
        
        logger.info("FORMAT_CONTEXT: Context compilation completed successfully.")
        return formatted_context
    except Exception as e:
        logger.error(f"FORMAT_CONTEXT: Error assembling text block context: {e}")
        return ""

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Retrieving pipeline started at {start_time} ...")
        
        client = create_chroma_client()
        collection = get_or_create_collection(client, config.COLLECTION_NAME)
        model = load_embedding_model()
        
        retriever_bundle = create_retriever(collection, model)
        
        query = "What was Apple's total revenue in 2023?"
        search_filter = {"ticker": "aapl", "year": 2023}
        
        chunks = hybrid_retrieve(retriever_bundle, query, filters=search_filter)
        context_output = format_context(chunks)
        print(context_output)
        
        end_time = datetime.now()
        logger.info(f"Retrieving pipeline ended at {end_time}, execution duration: {end_time - start_time} ...")
    except Exception as e:
        logger.error(f"Retrieving pipeline failed: {e}")