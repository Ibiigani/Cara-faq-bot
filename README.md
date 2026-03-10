# Cara-faq-bot
RAG chatbot for CARA using LangChain, ChromaDB, HuggingFace and Streamlit

This chatbot allows users to ask questions about CARA(Center for Alumni Relations & Advancement at South Dakota School of Mines and Technology). It retrieves relevant information from a pdf  documents and generates answers using a local LLM.

---

## Features

- Automatic document embedding with **HuggingFace sentence-transformers**
- Vector search using **ChromaDB**
- Question answering with **local LLM** (Flan-T5)
- Streamlit-based interactive web interface
- Persistent vector database for repeated queries

---
## Installation

## Clone the repository

---
git clone https://github.com/Ibiigani/Cara-faq-bot.git
cd Cara-faq-bot

##  Create and activate a virtual environment

python -m venv myenv
.\myenv\Scripts\Activate   # Windows
# or
source myenv/bin/activate  # macOS/Linux

## Install dependencies

pip install -r requirements.txt
## Usage
Step 1 – Ingest document

python ingest.py

This will create vector embeddings that is saved in db.

Step 2 – Run the chatbot
streamlit run chatbot.py

Open the Streamlit link in your browser.
Ask a question about CARA



## Technologies Used

LangChain – RAG pipeline

ChromaDB – Vector database

HuggingFace Transformers – Local LLM

Streamlit – Web app interface

Python 3.10+
## db folder
Note: The db/ folder is not included in the repository because it is generated automatically by running ingest.py.