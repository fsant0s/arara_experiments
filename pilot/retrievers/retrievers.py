
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_BASE_PATH = "../datasets/mal_2023/1000/chrome_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

retriever_users = Chroma(
    collection_name="users",
    persist_directory=os.path.join(CHROMA_BASE_PATH, "users"),
    embedding_function=embeddings
).as_retriever(search_kwargs={"k": 100})

retriever_anime = Chroma(
    collection_name="animes",
    persist_directory=os.path.join(CHROMA_BASE_PATH, "animes"),
    embedding_function=embeddings
).as_retriever(search_kwargs={"k": 100})

retriever_animes_users = Chroma(
    collection_name="animes_users",
    persist_directory=os.path.join(CHROMA_BASE_PATH, "animes_users"),
    embedding_function=embeddings
).as_retriever(search_kwargs={"k": 100})