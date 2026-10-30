import streamlit as st
import logging
import json
import re
import sys
from pathlib import Path

# Setup absolute path imports
parent_dir = str(Path(__file__).resolve().parent)
sys.path.append(parent_dir)

import config

# Import pipeline components
from src.indexing.embedder import load_embedding_model
from src.indexing.vector_store import create_chroma_client, get_or_create_collection
from src.retrieval.retriever import create_retriever, hybrid_retrieve, format_context
from src.retrieval.reranker import load_reranker, rerank
from src.generation.generator import load_groq_client, generate

logger = logging.getLogger(__name__)

# Company Name to Ticker mapping dictionary
COMPANY_LOOKUP = {
    "Apple Inc.": "aapl",
    "Tesla, Inc.": "tsla",
    "Microsoft Corporation": "msft",
    "Amazon.com, Inc.": "amzn",
    "Alphabet Inc. (Google)": "googl"
}

# ==========================================
# 1. SYSTEM INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Booting AI Models & Vector Database...")
def initialize_system():
    logger.info("STREAMLIT: Initializing backend models...")
    client = create_chroma_client()
    collection = get_or_create_collection(client, config.COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError(f"The Chroma collection '{config.COLLECTION_NAME}' is empty.")
    model = load_embedding_model()
    retriever = create_retriever(collection, model)
    reranker = load_reranker()
    groq_client = load_groq_client()
    
    return retriever, reranker, groq_client


def setup_errors():
    errors = []
    for name in ("COLLECTION_NAME", "GROQ_API_KEY", "GROQ_MODEL"):
        if not getattr(config, name, None):
            errors.append(f"Set {name} in your environment or Streamlit secrets.")

    for name in ("CHUNKS_PATH", "CHROMA_DB_PATH"):
        value = getattr(config, name, None)
        if not value or not Path(value).is_dir():
            errors.append(f"Set {name} to an existing directory.")

    if config.CHROMA_DB_PATH and not (Path(config.CHROMA_DB_PATH) / "chroma.sqlite3").is_file():
        errors.append("The Chroma database file is missing from CHROMA_DB_PATH.")
    return errors


def source_labels(chunks):
    labels = []
    for chunk in chunks:
        metadata = chunk.get("metadatas") or chunk.get("metadata") or {}
        filename = chunk.get("source") or metadata.get("source")
        if not filename:
            filename = f"{metadata.get('ticker', 'unknown')}_{metadata.get('year', 'unknown')}_10k.pdf"
        page = metadata.get("page_number")
        label = f"{filename}, page {page}" if page is not None else filename
        if label not in labels:
            labels.append(label)
    return labels


def show_sources(sources):
    if sources:
        with st.expander("Sources used for this answer"):
            for source in sources:
                st.write(source)


def process_query_intent(chat_history, company_name, year, client):
    """
    Categorizes query intent (Chitchat, Out-of-Domain, or Financial)
    and formats a standalone RAG query anchored to the selected company and year.
    """
    sys_prompt = f"""You are a precise routing agent for a Financial SEC 10-K database. 

Current Active Target:
- Company Name: {company_name}
- Fiscal Year: {year}

Analyze the user's LATEST message in context of the conversation. Output ONLY raw JSON. Do not include markdown formatting.

CATEGORY 1: GREETINGS & CHITCHAT
If the message is a simple greeting ("hi", "hello") or casual small talk:
{{"is_financial_query": false, "is_out_of_domain": false, "response": "Hello! I am ready to answer your questions regarding {company_name}'s {year} SEC 10-K filing. What would you like to know?"}}

CATEGORY 2: OUT OF DOMAIN
If the message asks about topics completely unrelated to corporate finance, stocks, or SEC filings (e.g. recipes, sports, coding help, weather):
{{"is_financial_query": false, "is_out_of_domain": true, "response": "I am a specialized financial assistant. I can only answer questions related to corporate SEC 10-K filings, revenue, and business risks."}}

CATEGORY 3: FINANCIAL QUERY
If the message is asking a financial/corporate question:
Rephrase the user's latest query into a clear standalone search query explicitly incorporating {company_name} and {year}.
{{"is_financial_query": true, "is_out_of_domain": false, "standalone_query": "What was {company_name}'s total revenue in {year}?"}}
"""

    messages = [{"role": "system", "content": sys_prompt}]
    
    for msg in chat_history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=0,
            stream=False 
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json\n", "", 1).strip()
            
        intent_data = json.loads(raw_text)
        return intent_data
        
    except Exception as e:
        logger.error(f"Intent Processing Failed: {e}")
        return None

# ==========================================
# 2. UI CONFIGURATION & SIDEBAR
# ==========================================
st.set_page_config(page_title="FilingLens", page_icon="📊", layout="centered")

st.title("FilingLens")
st.markdown(
    "Explore SEC 10-K annual reports by asking questions about revenue, cash flow, "
    "business risks, and more. Choose a company and fiscal year in the sidebar; "
    "financial answers include the filing pages used."
)

# Sidebar Selection for Company and Year
with st.sidebar:
    st.header("🎯 Search Scope")
    
    available_companies = list(COMPANY_LOOKUP.keys())
    selected_company_name = st.selectbox("Company Name", options=available_companies)
    selected_ticker = COMPANY_LOOKUP[selected_company_name]
    
    selected_year = st.selectbox("Fiscal Year", options=config.YEARS)
    
    st.divider()
    st.caption(f"**Active Target:** `{selected_ticker.upper()}` ({selected_year})")

# Check required settings and data before loading the models.
errors = setup_errors()
if errors:
    st.error("App setup is incomplete:\n\n" + "\n".join(f"- {error}" for error in errors))
    st.stop()

try:
    retriever_bundle, reranker_model, groq_client = initialize_system()
except Exception:
    logger.exception("Unable to initialize the financial assistant")
    st.error("Could not load the search index or AI models. Check the app logs and model downloads.")
    st.stop()

# ==========================================
# 3. CHAT INTERFACE & STATE MANAGEMENT
# ==========================================
current_scope = (selected_ticker, selected_year)
if st.session_state.get("active_scope") != current_scope:
    st.session_state.active_scope = current_scope
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

# Capture user input
if raw_user_input := st.chat_input(f"Ask anything about {selected_company_name} ({selected_year})..."):
    
    st.session_state.messages.append({"role": "user", "content": raw_user_input})
    with st.chat_message("user"):
        st.markdown(raw_user_input)

    with st.chat_message("assistant"):
        # Handle simple greetings without using the hosted model.
        if re.fullmatch(r"\s*(?:hi|hello|hey|good morning|good afternoon|good evening)[!.,?\s]*", raw_user_input, re.IGNORECASE):
            response_text = (
                f"Hello! Ask me about {selected_company_name}'s {selected_year} SEC 10-K filing."
            )
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.stop()

        # Route questions that need the hosted model.
        intent_data = process_query_intent(
            st.session_state.messages, 
            selected_company_name, 
            selected_year, 
            groq_client
        )
        
        if not intent_data:
            error_msg = "I encountered an error understanding that request. Please try again."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()

        # Step B: Handle Chitchat OR Out-of-Domain instantly (No spinner rendered!)
        if not intent_data.get("is_financial_query"):
            response_text = intent_data.get("response", "I am a financial assistant and cannot help with that topic.")
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.stop()
            
        # Step C: Prepare RAG search parameters
        search_filter = {"ticker": selected_ticker, "year": int(selected_year)}
        rag_query = intent_data.get("standalone_query", raw_user_input)

        filing_chunks = Path(config.CHUNKS_PATH) / f"{selected_ticker}_{selected_year}_chunks.json"
        if not filing_chunks.is_file():
            response_text = f"I don't have the 10-K filing for **{selected_company_name}** ({selected_year}) in the database yet."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.stop()
        
        # Step D: Only trigger the filing paper spinner when actually searching files!
        with st.spinner("Looking through filing papers..."):
            try:
                # Step E: Hybrid Retrieval
                chunks = hybrid_retrieve(retriever_bundle, rag_query, filters=search_filter)
                
                if not chunks:
                    response_text = "I couldn't find relevant passages in this filing for that question."
                    st.markdown(response_text)
                    sources = []
                else:
                    # Step F: Rerank, Format, and Generate
                    reranked_chunks = rerank(rag_query, chunks, reranker_model)
                    context = format_context(reranked_chunks)
                    
                    response_text = generate(rag_query, context, groq_client)
                    st.markdown(response_text)
                    sources = source_labels(reranked_chunks)
                    show_sources(sources)

                st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": sources})

            except Exception:
                error_msg = "I encountered an error while processing that request. Please try again."
                st.error(error_msg)
                logger.exception("APP ERROR: Query processing failed")
