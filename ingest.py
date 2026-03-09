import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# folder where documents will be stored
DOCS_FOLDER = "docs"

# folder to save vector db
DB_FOLDER = "db"

# list to hold all loaded documents
documents = []

# loop through files in docs/
for file in os.listdir(DOCS_FOLDER):
    filepath = os.path.join(DOCS_FOLDER, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(filepath)
        documents.extend(loader.load())

if not documents:
    print("No documents found")
    exit(1)

# local embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create chroma vector store
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_FOLDER
)

# persist database to disk
vectordb.persist()

print("Documents embedded successfully")
