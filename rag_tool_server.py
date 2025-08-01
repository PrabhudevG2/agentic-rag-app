# file: rag_tool_server.py
import chromadb
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
import time # Import the time library

# --- Configuration & Setup ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

mcp_server = FastMCP(name="RAGToolServer")

class PDFToolInput(BaseModel):
    query: str = Field(description="The question to be answered by searching the PDF document.")

# --- KEY CHANGE: Add timing logs to the startup process ---
print("[RAG Server] Script started.")
start_time = time.time()

print("[RAG Server] Loading embedding model... (This may take a moment)")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
end_time = time.time()
print(f"[RAG Server] Embedding model loaded in {end_time - start_time:.2f} seconds.")

print("[RAG Server] Connecting to ChromaDB...")
db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = db_client.get_collection(name=COLLECTION_NAME)
print("[RAG Server] RAG server components fully loaded and ready.")

@mcp_server.tool
async def answer_pdf_question(args: PDFToolInput, ctx: Context) -> str:
    """
    Answers a question by searching a technical PDF document.
    """
    query = args.query
    await ctx.info(f"PDF tool received query: '{query}'")
    
    await ctx.info("Generating query embedding...")
    query_embedding = embedding_model.encode(query).tolist()
    
    await ctx.info("Querying vector store...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    if not results['documents'] or not results['documents'][0]:
        await ctx.info("Query returned no results from the vector store.")
        return "No relevant information was found in the document for that query."

    doc_chunks = results['documents'][0]
    valid_chunks = [chunk for chunk in doc_chunks if isinstance(chunk, str)]

    if not valid_chunks:
        await ctx.info("Query found chunks, but they were not valid strings.")
        return "Found potential matches in the document, but could not retrieve content."

    context = "\n---\n".join(valid_chunks)
    
    await ctx.info("Found relevant context from PDF.")
    return f"Retrieved context from PDF: {context}"

if __name__ == "__main__":
    print("Starting RAG Tool Server at http://localhost:8002/mcp")
    mcp_server.run(transport="http", host="0.0.0.0", port=8002, path="/mcp")