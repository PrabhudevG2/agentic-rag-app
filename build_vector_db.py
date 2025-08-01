# file: build_vector_db.py
# (Or whatever your indexing script is named)

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
PDF_PATH = "/Users/prabhudev.guntur/Downloads/FORMULATION & EVALUATION OF WOUND HEALING.pdf" # Make sure this path is correct
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("--- Starting RAG Indexing Process ---")
    
    # 1. Load the document
    print(f"Loading text from '{PDF_PATH}'...")
    loader = PyPDFLoader(PDF_PATH)
    doc = loader.load()
    # PyPDFLoader returns a list of documents, one per page. We join them.
    full_document_text = "\n".join([page.page_content for page in doc])
    print(f"Successfully extracted {len(full_document_text)} characters.")
    
    # 2. Chunk the document text
    print("Chunking document text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(full_document_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # 3. Load the embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4. Setup the vector store
    print(f"Setting up vector store at '{CHROMA_DB_PATH}'...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # If the collection exists, delete it for a fresh build
    if COLLECTION_NAME in [c.name for c in db_client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it for a fresh build.")
        db_client.delete_collection(name=COLLECTION_NAME)

    collection = db_client.create_collection(name=COLLECTION_NAME)

    # 5. Generate embeddings and store them
    print("Generating embeddings and storing chunks in the vector store...")

    # --- THIS IS THE CRITICAL FIX ---
    # We must pass the actual text chunks to the 'documents' parameter.
    
    # Create simple metadata (optional, but good practice)
    chunk_metadatas = [{"source": PDF_PATH, "chunk_num": i} for i in range(len(text_chunks))]
    
    # Create unique IDs for each chunk
    chunk_ids = [f"chunk_{i}" for i in range(len(text_chunks))]

    # Add to the collection correctly
    collection.add(
        documents=text_chunks,  # The list of text strings goes here
        ids=chunk_ids,
        metadatas=chunk_metadatas
    )

    print("\n--- RAG Indexing Process Complete ---")
    print(f"Vector store created with {collection.count()} items.")

if __name__ == "__main__":
    main()