from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils import load_config
import chromadb

config = load_config()

def get_ollama_embeddings():
    return OllamaEmbeddings(model=config["ollama"]["embedding_model"], base_url=config["ollama"]["base_url"])

def load_vectordb(embeddings=get_ollama_embeddings()):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma