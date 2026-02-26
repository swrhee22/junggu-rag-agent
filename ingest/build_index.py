import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_DIR = Path("data/pdf")
DB_DIR = "db/chroma"

def load_pdfs(pdf_dir: Path):
    docs = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        # 파일명 메타데이터 추가
        for d in pdf_docs:
            d.metadata["source_file"] = pdf_path.name
        docs.extend(pdf_docs)
    return docs

def main():
    docs = load_pdfs(PDF_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="junggu_guides"
    )
    print(f"Done. docs={len(docs)}, chunks={len(chunks)}, db={DB_DIR}")

if __name__ == "__main__":
    main()