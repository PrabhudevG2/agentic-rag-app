# file: test_rag_tool.py
import asyncio
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration (must match rag_tool_server.py) ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

async def main():
    """
    A standalone script to test queries directly against the ChromaDB vector store.
    """
    print("--- RAG Tool Standalone Test ---")
    
    try:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print(f"Connecting to ChromaDB at '{CHROMA_DB_PATH}'...")
        db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"Successfully connected to collection '{COLLECTION_NAME}' with {collection.count()} items.")
        print("-" * 30)

    except Exception as e:
        print(f"\n--- FATAL ERROR during setup: {e}")
        return

    # --- Main Test Loop ---
    while True:
        try:
            user_query = input("\nEnter a query to test (or 'exit' to quit): \n> ")
            if user_query.lower() == "exit":
                break

            print("\n1. Generating query embedding...")
            query_embedding = embedding_model.encode(user_query).tolist()
            
            print("2. Querying vector store...")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5  # Let's get 5 results to see more
            )
            
            print("\n--- RAW RESULTS ---")
            print(results)
            print("-------------------\n")

            print("3. Processing results...")
            if not results['documents'] or not results['documents'][0]:
                print("\n[CONCLUSION] No document chunks were found for this query.")
                continue

            doc_chunks = results['documents'][0]
            valid_chunks = [chunk for chunk in doc_chunks if isinstance(chunk, str)]

            if not valid_chunks:
                print("\n[CONCLUSION] Chunks were found, but none of them were valid strings.")
                continue
            
            context = "\n---\n".join(valid_chunks)
            print("\n[CONCLUSION] Successfully retrieved the following context:\n")
            print(context)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting test script.")