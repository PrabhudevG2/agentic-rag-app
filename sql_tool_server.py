# file: sql_tool_server.py
import sqlite3
import os
import re
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# --- LangChain Imports for Structured Output ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# --- Configuration & Setup ---
DB_FILE = "company.db"
load_dotenv()

mcp_server = FastMCP(name="SQLToolServer")

# --- Pydantic Models ---
# 1. Model for the tool's input (from the agent)
class DatabaseToolInput(BaseModel):
    query: str = Field(description="The natural language question to be converted into an SQL query.")

# 2. Model for the structured output we expect from the LLM
class SqlQuery(BaseModel):
    query: str = Field(description="The generated SQLite query.")

# --- Helper Function (no changes) ---
def get_db_schema(db_path: str) -> str | None:
    if not os.path.exists(db_path): return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schema_rows = cursor.fetchall()
        conn.close()
        return "\n".join([row[0] for row in schema_rows if row[0]])
    except Exception as e:
        print(f"Error reading database schema: {e}")
        return None

# --- Main Tool Function (Refactored for Structured Output) ---
@mcp_server.tool
async def answer_database_question(args: DatabaseToolInput, ctx: Context) -> str:
    """
    Answers a question by dynamically generating and executing an SQL query on the company database.
    The database contains tables about employees, products, and sales.
    """
    user_query = args.query
    await ctx.info(f"Database tool received query: '{user_query}'")

    # --- Initialize components for the structured output chain ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    output_parser = JsonOutputParser(pydantic_object=SqlQuery)
    
    db_schema = get_db_schema(DB_FILE)
    if not db_schema:
        await ctx.error(f"Database file '{DB_FILE}' not found or schema is unreadable.")
        return f"Error: Database file '{DB_FILE}' not found."

    # Create the prompt template, including the JSON format instructions from the parser
    prompt = PromptTemplate(
        template="""You are an expert SQLite data analyst. Based on the database schema, convert the user's question into a single, executable SQLite query.
        \n{format_instructions}\n
        Schema:
        {schema}
        
        Question:
        {query}""",
        input_variables=["query", "schema"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    # Create the full generation chain
    chain = prompt | llm | output_parser

    await ctx.info("Generating SQL query with LLM (expecting JSON)...")
    try:
        # Invoke the chain to get a structured dictionary
        response_dict = await chain.ainvoke({"query": user_query, "schema": db_schema})
        generated_sql = response_dict['query']
        await ctx.info(f"LLM Generated SQL (from JSON): {generated_sql}")

    except Exception as e:
        await ctx.error(f"Error during structured SQL generation: {e}")
        return f"Error during SQL generation: {e}"

    # We can now trust the format of generated_sql, but we still check it for security.
    if not generated_sql.lower().strip().startswith("select"):
        await ctx.error("Generated query was not a 'SELECT' statement. Aborting for security.")
        return f"Error: Invalid query generated. Must be a 'SELECT' statement. The LLM returned: {generated_sql}"

    await ctx.info(f"Executing SQL query: {generated_sql}")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        results = cursor.fetchall()
        conn.close()

        if not results:
            await ctx.info("Query executed successfully, but returned no results.")
            return "Query executed successfully, but returned no results."

        column_names = [description[0] for description in cursor.description]
        formatted_results = f"Query Result:\nColumns: {', '.join(column_names)}\n" + "\n".join(map(str, results))
        
        await ctx.info("Query executed successfully, returning formatted results.")
        return formatted_results.strip()
    except sqlite3.Error as e:
        await ctx.error(f"Database query failed: {e}")
        return f"Database query failed with error: {e}\nAttempted Query: {generated_sql}"


if __name__ == "__main__":
    print("Starting SQL Tool Server at http://localhost:8001/mcp")
    mcp_server.run(transport="http", host="0.0.0.0", port=8001, path="/mcp")