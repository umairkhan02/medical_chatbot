from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY="pcsk_3DRwuU_SMtysDinrW9Rwe4g9mcopgKxjb3QQ4pHNYqym4oYUZ5QKnzJLAhWHtyJCNwqHjS"
OPENAI_API_KEY = "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"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_docs = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_docs)
text_chunks = text_split(filter_data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)