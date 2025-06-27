from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data = 'Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Pinecone and Create Index
pc = Pinecone(api_key = "pcsk_6b7g47_AWy5TkuhPy1eymv4zEBLvngC9BgGv8ehkqRBPJGkt9CCiXmseKcTAndik7wSmzE")

index_name = "medicalbot"

pc.create_index(
    name = index_name,
    dimension = 384,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        )
)


# Pinecone Vector Storing
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings
)