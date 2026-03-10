from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# configuration
PDF_FILE = "docs/Cara_faq .pdf"
DB_FOLDER = "db"

# load the PDF
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()

# initiate the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# initiate vector database
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_FOLDER
)

print("Document embedded successfully")