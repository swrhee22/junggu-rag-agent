from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "db/chroma"
COLLECTION_NAME = "junggu_guides"

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever