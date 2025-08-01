# Agentic RAG: SQL & PDF Search

This project demonstrates a sophisticated, tool-based agentic application that can answer user queries by retrieving information from two distinct sources: a structured SQL database and an unstructured PDF document. It provides two different implementations of the core agent logic: one using LangGraph for stateful, graph-based control flow, and another using CrewAI for a collaborative, multi-agent approach.

## Features

- **Dual-Source RAG**: Seamlessly answers questions from either a SQL database or a PDF, using the appropriate tool for the job.
- **Modular Tool Architecture**: SQL and PDF search functionalities are exposed as independent servers using FastMCP, making the system modular and scalable.
- **Choice of Agent Frameworks**:
  - LangGraph (`agent.py`): Implements a robust, state-machine-based agent that explicitly manages the flow between the user, LLM, and tools.
  - CrewAI (`crewai_agent.py`): Implements a hierarchical crew with a manager agent delegating tasks to specialized agents for database analysis and document research.
- **LLM Integration**: Powered by Google's Gemini family of models.
- **Easy Setup**: Includes simple scripts to create the sample database and build the vector store from your PDF.
- **Debugging & Tracing**: Offers optional, easy-to-configure integration with LangSmith for observing and debugging agent behavior.

## Core Architecture: The MCP Setup

A key feature of this project's architecture is the use of the Multi-Agent Communication Protocol (MCP) to create a modular and robust system. Instead of running all the code in a single, monolithic script, we separate the core agent logic from the tools it uses.

MCP acts as the communication layer that allows the main agent (the "client") to discover and execute tools running on separate, independent servers. This project uses the `fastmcp` library to create the tool servers and `langchain-mcp-adapters` to connect the agent to them.

### 1. The Tool Servers (FastMCP)

The specialized tools for searching the SQL database and the PDF document are not just functions; they are services running as standalone web servers. This is achieved using FastMCP.

- **sql_tool_server.py**: This script launches a web server (on port 8001 by default) that listens for requests.
  - The `@mcp_server.tool` decorator exposes the `answer_database_question` function as a callable tool over the network.
  - When the agent needs to query the database, it sends a request to this server. The server receives the request, generates and executes the SQL, and sends the result back.

- **rag_tool_server.py**: Similarly, this script launches a server (on port 8002) that exposes the `answer_pdf_question` tool.
  - It handles all the logic for embedding the user's query and searching the Chroma vector database.

#### Why is this useful?

- **Separation of Concerns**: The agent doesn't need to know how the SQL is generated or how the PDF is searched. It only needs to know that a tool exists to do the job.
- **Scalability**: If our PDF search tool became very resource-intensive, we could move it to a more powerful machine without changing the agent code at all.
- **Modularity**: We can add, remove, or update tools without ever touching or restarting the main agent application.

### 2. The Agent Client (MultiServerMCPClient & MCPServerAdapter)

The main agent files (`agent.py` and `crewai_agent.py`) act as the clients that consume the tools offered by the MCP servers.

#### In `agent.py` (LangGraph)

The LangGraph implementation uses the `MultiServerMCPClient` to connect to the tool servers during setup.

```python
# file: agent.py
client = MultiServerMCPClient(
    {
        "sql_server": {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable_http"},
        "rag_server": {"url": "http://127.0.0.1:8002/mcp", "transport": "streamable_http"},
    }
)
tools = await client.get_tools()
```

This client connects to both URLs, collects all the tools advertised by them (`answer_database_question` and `answer_pdf_question`), and makes them available as a single list of tools that can be bound to the LLM.

#### In `crewai_agent.py` (CrewAI)

The CrewAI implementation uses the `MCPServerAdapter`, which is a convenient context manager (`with ... as`) for the same purpose.

```python
# file: crewai_agent.py
server_params_list = [
    {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable-http"}, # SQL Tool
    {"url": "http://127.0.0.1:8002/mcp", "transport": "streamable-http"}, # PDF Tool
]
with MCPServerAdapter(server_params_list) as mcp_tools:
    company_db_tool = mcp_tools["answer_database_question"]
    pdf_search_tool = mcp_tools["answer_pdf_question"]
```

The adapter fetches the tools from the servers, and we can assign them directly to the specialized agents who need them. The Database Analyst gets the `company_db_tool`, and the Document Researcher gets the `pdf_search_tool`.




## How It Works (Overall Flow)

The application follows a tool-based agent architecture. The user's query is first assessed by a primary agent (or a manager agent in CrewAI). This agent then decides which specialized tool is best suited to answer the question.

1. **User Query**: The user asks a question like "Who sold the most keyboards?" or "What is the formulation for wound healing?".
2. **Agent Brain (LLM)**: The main agent (either LangGraph or CrewAI's manager) analyzes the query.
3. **Tool Selection**: Based on the analysis, it selects either the `CompanyDatabaseTool` for SQL-related questions or the `PDFDocumentSearchTool` for PDF-related questions.
4. **Tool Execution**: The agent calls the selected tool, which is running as a separate MCP server.
   - The SQL Tool Server receives the query, uses an LLM to generate a safe SELECT SQL query, executes it against the `company.db`, and returns the result.
   - The RAG Tool Server receives the query, creates a vector embedding, searches the Chroma vector database for relevant text chunks from the PDF, and returns the context.
5. **Final Response**: The agent receives the data from the tool and uses the LLM to formulate a final, user-friendly answer.


## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/PrabhudevG2/agentic-rag-app
cd agentic-rag-app
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv
# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

All required Python packages are listed in the `requirements_freezed.txt` file.

```bash
pip install -r requirements_freezed.txt
```

### Step 4: Configure Environment Variables

You need to provide API keys for the services used in this project. Create a file named `.env` in the root of the project directory.

```bash
touch .env
```

Now, open the `.env` file and add the following content. You must provide a `GOOGLE_API_KEY`.

```env
# --- Google Gemini Configuration (Required) ---
# Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
# --- LangSmith Configuration (Optional) ---
# Uncomment these lines to enable tracing for debugging.
# Get your key from https://smith.langchain.com/
# LANGCHAIN_TRACING="true"
# LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY_HERE"
# LANGCHAIN_PROJECT="My Agentic RAG App" # You can change this project name
```

### Step 5: Customize Data Sources (Important!)

#### A. To Use Your Own PDF

- Place your PDF file in a known location.
- Open the `build_vector_db.py` file.
- Change the `PDF_PATH` variable to the correct path of your PDF file.

```python
# file: build_vector_db.py
...
# --- Configuration ---
PDF_PATH = "/path/to/your/document.pdf" # <-- CHANGE THIS
...
```

#### B. To Modify the SQL Database

- Open the `setup_database.py` file.
- You can change the table schemas (`CREATE TABLE` statements) and the data (`INSERT INTO` statements) to match your own data.
- Note: If you significantly change the table or column names, you may need to adjust the system prompts in `agent.py` and `crewai_agent.py` to give the LLM better context about your new schema.

## Running the Application

The application consists of multiple services that must be run simultaneously. You will need to open three separate terminal windows. Make sure you have activated the virtual environment (`source venv/bin/activate`) in each terminal.

### Step 1: Set Up the Data (Run this once)

First, prepare the SQL database and the vector store for your PDF.

```bash
# Create the SQLite database
python setup_database.py
# Process your PDF and create the vector store
# This may take a minute depending on the PDF size
python build_vector_db.py
```

### Step 2: Start the Tool Servers

These servers listen for requests from the main agent.

➡️ In your 1st Terminal:

```bash
# Start the SQL tool server
python sql_tool_server.py
```

➡️ In your 2nd Terminal:

```bash
# Start the RAG (PDF search) tool server
python rag_tool_server.py
```

### Step 3: Run the Agent

Choose one of the agent implementations to run.

➡️ In your 3rd Terminal (Option A - LangGraph):

```bash
# Run the LangGraph-based agent
python agent.py
```

➡️ In your 3rd Terminal (Option B - CrewAI):

```bash
# Run the CrewAI-based agent
python crewai_agent.py
```

or you can run everything using below



The application consists of multiple services that must be run simultaneously. You can use the `run.sh` script to simplify this process. Make sure you have activated the virtual environment (`source venv/bin/activate`).

### Using `run.sh`

The `run.sh` script automates the process of setting up and running the application. It performs the following steps:

1. **Choose Agent Framework**: You can specify the agent framework to use (`langgraph` or `crewai`). The default is `langgraph`.
2. **Kill Existing Server Processes**: It finds and stops any old server processes that might be running.
3. **Conditionally Setup Databases**: It checks for the existence of the SQLite database and ChromaDB directory and sets them up if they are not found.
4. **Start Servers and Wait**: It starts the SQL and RAG tool servers in the background and waits for them to be ready.
5. **Start the Main Agent Application**: Finally, it starts the main agent application using the specified framework.

To run the script, use the following command:

```bash
./run.sh
```
If you want to use the CrewAI framework, you can specify it as an argument:

```bash
./run.sh crewai
```

## Usage

Once the agent is running, you can start asking questions in the terminal where you launched `agent.py` or `crewai_agent.py`.

### Example SQL Queries:

- How many employees are in the engineering department?
- List all products and their prices.
- Who is the top-performing sales employee based on the number of sales?

### Example PDF Queries:

(These depend on the content of your PDF. The example below assumes the PDF is about wound healing.)

- What does the document say about wound healing formulation?
- Summarize the introduction of the document.
- What are the evaluation parameters?

### To Exit:

Type `exit` and press Enter.

## File Breakdown

- `agent.py`: (Option 1) Main application entry point using the LangGraph framework.
- `crewai_agent.py`: (Option 2) Alternative application entry point using the CrewAI framework.
- `sql_tool_server.py`: A FastMCP server that exposes the SQL database query tool.
- `rag_tool_server.py`: A FastMCP server that exposes the PDF vector search tool.
- `setup_database.py`: A utility script to create and populate the `company.db` SQLite database.
- `build_vector_db.py`: A utility script to read a PDF, chunk it, and store its embeddings in a ChromaDB vector store.
- `requirements_freezed.txt`: A list of all Python dependencies for the project.
- `.env`: (You create this) Stores your secret API keys and configuration.
