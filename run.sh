#!/bin/bash

# This script finds and kills old processes, then restarts the application.
# It now checks for databases and allows choosing the agent framework.

# --- Step 0: Choose Agent Framework ---
FRAMEWORK="langgraph" # Default
if [ "$1" == "crewai" ]; then
    FRAMEWORK="crewai"
fi
echo "--- Using framework: $FRAMEWORK ---"


# --- Step 1: Kill existing server processes ---
echo "--- Finding and stopping any old server processes... ---"
pkill -f "sql_tool_server.py" && echo "Stopped old SQL server." || echo "SQL server was not running."
pkill -f "rag_tool_server.py" && echo "Stopped old RAG server." || echo "RAG server was not running."
echo "--- Waiting for ports to be released... ---"
sleep 2

# --- Step 2: Conditionally Setup Databases ---
echo "--- Checking for SQLite database... ---"
if [ ! -f "company.db" ]; then
    echo "SQLite database 'company.db' not found. Running setup..."
    python3 setup_database.py
else
    echo "SQLite database 'company.db' already exists. Skipping setup."
fi

echo -e "\n--- Checking for Vector Database... ---"
if [ ! -d "chroma_db" ]; then
    echo "ChromaDB directory 'chroma_db' not found. Building vector database from PDF..."
    python3 build_vector_db.py
    if [ $? -ne 0 ]; then
        echo "Vector DB build failed. Aborting."
        exit 1
    fi
else
    echo "ChromaDB directory 'chroma_db' already exists. Skipping build."
fi

# Define a cleanup function
cleanup() {
    echo -e "\n--- Shutting down all background servers... ---"
    kill 0
    echo "All processes stopped."
}
trap cleanup INT

# --- Step 3: Start Servers and Wait ---
echo -e "\n--- Starting Tool Servers in the background (logs in sql_server.log & rag_server.log) ---"
python3 sql_tool_server.py > sql_server.log 2>&1 &
python3 rag_tool_server.py > rag_server.log 2>&1 &

wait_for_port() {
    local port=$1; local name=$2
    echo -n "Waiting for $name on port $port..."
    while ! nc -z localhost $port; do
        sleep 0.5 && echo -n ".";
    done
    echo " $name is up!"
}
wait_for_port 8001 "SQL Tool Server"
wait_for_port 8002 "RAG Tool Server"

# --- Step 4: Start the Main Agent Application ---
echo -e "\n--- Servers are up. Starting the Main Agent Application ---"
if [ "$FRAMEWORK" == "crewai" ]; then
    python3 crewai_agent.py
else
    python3 agent.py # Default to the langgraph agent
fi

# When the agent exits, call the cleanup function
cleanup