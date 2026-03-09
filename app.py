import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="CARA FAQ Bot")
st.title("CARA FAQ Assistant")

# ----------------------------
# Chat History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Load Embeddings (LOCAL)
# ----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ----------------------------
# Load Vector DB
# ----------------------------
@st.cache_resource
def load_vectordb():
    return Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

vectordb = load_vectordb()

# ----------------------------
# Retriever (most relevant chunks only)
# ----------------------------
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 5}  # only top chunk
)

# ----------------------------
# Load Local LLM (Transformers 5.3 compatible)
# ----------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",  # v5 replacement for text2text-generation
        model="google/flan-t5-base",
        max_new_tokens=120,
        temperature=0.2
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ----------------------------
# Display Chat History
# ----------------------------
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)

# ----------------------------
# User Input
# ----------------------------
query = st.chat_input("Ask a question about CARA")

# ----------------------------
# RAG Pipeline
# ----------------------------
if query:

    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching CARA FAQ..."):

            # Retrieve top chunks
            docs = retriever.invoke(query)

            if not docs:
                answer = "I couldn't find that in the CARA FAQ."
            else:
                # Truncate each chunk to avoid long dump
                context = "\n\n".join([doc.page_content[:300] + "..." for doc in docs])

                prompt = f"""
Context:
{context}
"""

                response = llm.invoke(prompt)

                # Handle Transformers v5 response format
                if isinstance(response, list):
                    answer = response[0].get("generated_text", "")
                elif isinstance(response, dict):
                    answer = response.get("text", "")
                else:
                    answer = str(response)

            st.write(answer)

    st.session_state.history.append(("assistant", answer))

