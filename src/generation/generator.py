from openai import OpenAI
from pathlib import Path
import sys
import logging
from datetime import datetime
import time


parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)

import config
from src.indexing.embedder import load_embedding_model
from src.indexing.vector_store import create_chroma_client, get_or_create_collection
from src.retrieval.retriever import create_retriever, retrieve, bm25_retrieve,format_context
from src.retrieval.reranker import load_reranker, rerank

LOGS_PATH=config.LOGS_PATH
NVIDIA_MODEL=config.NVIDIA_MODEL

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def load_nvidia_client():
    try:
        logger.info(f"LOAD_NVIDIA_CLIENT: Creating connection with {NVIDIA_MODEL} model in process ...")

        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = config.NVIDIA_API_KEY
        )

        logger.info(f"LOAD_NVIDIA_CLIENT: Connection with {NVIDIA_MODEL} model has established successfully ...")

        return client
    except Exception as e:
        logger.error(f"LOAD_NVIDIA_CLIENT: Error: {e}")
        raise

def build_prompt(query, context):
    try:
        logger.info(f"BUILD_PROMPT: Building system prompt based on user query: {query} and relevent context ...")

        SYSTEM_PROMPT = "You are a financial analyst assistant.\
            Answer the user's question using ONLY the provided context from SEC 10-K filings.\
            If the answer is not in the context, say : I don't have enough information to answer this question."
        final_prompt = SYSTEM_PROMPT + f"\n\nContext:\n{context}"

        return final_prompt
    except Exception as e:
        logger.error(f"BUILD_PROMPT: Error: {e}")
        raise

def generate(query, context, client, max_retries=5):
    try:
        logger.info(f"GENERATE: Generating answer based on user query ...")
        prompt = build_prompt(query, context)

        for attempt in range(max_retries):
            try:
                generated_answer = client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0,
                    stream=True
                )
                
                reasoning_chunks = []
                content_chunks = []

                for chunk in generated_answer:
                    if not chunk.choices:
                        continue
                    
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                        
                    if chunk.choices[0].delta.content is not None:
                        content_chunks.append(chunk.choices[0].delta.content)

                final_reasoning = "".join(reasoning_chunks).strip()
                final_answer = "".join(content_chunks).strip()

                logger.info(f"GENERATE: Stream complete. Reasoning length: {len(final_reasoning)} chars | Answer length: {len(final_answer)} chars.")
                return final_answer

            except Exception as api_error:
                error_msg = str(api_error)
                if "ResourceExhausted" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s
                        logger.warning(f"GENERATE: NVIDIA API congested. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                
                logger.error(f"GENERATE: Unrecoverable API Error: {api_error}")
                raise api_error

        logger.error("GENERATE: Max retries exhausted. The NVIDIA endpoint is overloaded.")
        return "I am currently unable to answer as the AI server is overloaded. Please try again in a few minutes."
        
    except Exception as e:
        logger.error(f"GENERATE: Pipeline Error: {e}")
        raise

def answer(query, filters, retriever, reranker, nvidia_client):
    try:
        logger.info(f"ANSWER: Generating user final answer in process ...")
        
        chunks = retrieve(retriever, query,filters=filters)
        reranked_chunks = rerank(query, chunks, reranker)
        context_output = format_context(reranked_chunks)
        final_answer = generate(query, context_output, nvidia_client)

        return final_answer
    except Exception as e:
        logger.error(f"ANSWER: Error: {e}")
        raise

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Generating pipeline started at {start_time} ...")
        
        client = create_chroma_client()
        collection = get_or_create_collection(client, config.COLLECTION_NAME)
        model = load_embedding_model()
        
        retriever_bundle = create_retriever(collection, model)
        reranker = load_reranker()
        nvidia_client = load_nvidia_client()
        
        query = "What was Apple's total revenue in 2023?"
        search_filter = {"ticker": "aapl", "year": 2023}

        final_answer = answer(query, search_filter, retriever_bundle, reranker, nvidia_client)
                
        end_time = datetime.now()
        logger.info(f"Generate pipeline ended at {end_time}, execution duration: {end_time - start_time} ...")
    except Exception as e:
        logger.error(f"Generate pipeline failed: {e}")