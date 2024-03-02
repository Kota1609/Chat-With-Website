import os
import json
import chromadb
from openai import OpenAI

import os
import warnings
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# from lib.embed import TextGenEmbeddings

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dbUNTurl")

def divide_chunks(lst, n):
    if n < 1:
        raise ValueError("Number of chunks must be a positive integer.")
        
    chunk_size, remainder = divmod(len(lst), n)
    chunks = []
    index = 0
    
    for i in range(n):
        current_chunk_size = chunk_size + (i < remainder)
        chunk = lst[index:index + current_chunk_size]
        chunks.append(chunk)
        index += current_chunk_size

    return chunks

pages = []
for file in os.listdir('data'):
    if file == 'visited.txt':
        continue
    try:
        data = { }
        with open(f'data/{file}', 'r') as f:
            data = json.load(f)
        pages.append(data)
    except:
        continue

import hashlib

documents = []
metadatas = []
ids = []
for page in pages:
    id = 0
    full_text = ""
    for obj in page["text_objects"]:
        if obj["tag"] != "p":
            full_text += "### "
        full_text += obj["text"] + '\n\n'
    for obj in page["text_objects"]:
        if obj["tag"] != "p":
            continue
        url = page["website_url"]
        hash = hashlib.md5((url + str(id)).encode()).hexdigest()
        if hash in ids:
            continue
        ids.append(hash)
        id += 1
        documents.append(obj["text"])
        metadatas.append({ "url": url, "clipping": obj["text"], "full_text": full_text })

doc_chunks = divide_chunks(documents, 100)
meta_chunks = divide_chunks(metadatas, 100)
id_chunks = divide_chunks(ids, 100)


def process_documents(ignored_files: List[str] = []) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    texts = text_splitter.split_documents(doc_chunks)
    return texts

def transform_documents(documents):
    transformed_documents = [
        Document(page_content=doc['full_text'], metadata={'source': doc['url']}) for doc in documents
    ]
    return transformed_documents

def create_vector_database():

    ollama_embeddings = OllamaEmbeddings(model="zephyr")
    texts = transform_documents(metadatas)

    vector_database = Chroma(
        # documents=chunked_documents,
        embedding_function=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    collection = vector_database.get()
    # texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    print(f"Creating embeddings. May take some minutes...")
    vector_database.add_documents(texts)


    vector_database.persist()

if __name__ == "__main__":
    create_vector_database()