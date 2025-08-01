# file: crewai_agent.py

import os
from dotenv import load_dotenv

# Load environment variables (must include GEMINI_API_KEY)
load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter

# --- Configuration and Setup ---
IS_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING") == "true" and os.getenv("LANGCHAIN_API_KEY")
GEMINI_API_KEY= os.getenv("GOOGLE_API_KEY")

def main():
    """
    Sets up and runs the CrewAI agent application using the official MCPServerAdapter.
    """
    print("--- CrewAI Agent using Gemini LLM ---")

    # Define MCP server parameters (update URLs if needed)
    server_params_list = [
        {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable-http"},  # SQL Tool
        {"url": "http://127.0.0.1:8002/mcp", "transport": "streamable-http"},  # PDF Tool
    ]

    try:
        with MCPServerAdapter(server_params_list) as mcp_tools:
            print(f"\nSuccessfully loaded tools: {[tool.name for tool in mcp_tools]}")

            company_db_tool = mcp_tools["answer_database_question"]
            pdf_search_tool = mcp_tools["answer_pdf_question"]

            # ✅ Use the official CrewAI LLM wrapper with Gemini
            llm = LLM(
                model="gemini/gemini-2.0-flash",
                temperature=0,
                api_key=GEMINI_API_KEY
            )

            # Define Specialist Agents
            print("Defining specialist agents...")

            database_analyst = Agent(
                role='Expert Database Analyst',
                goal='Answer user questions about company data by executing SQL queries.',
                backstory="You are a master of SQL who uses the `answer_database_question` tool to find information in the company database.",
                tools=[company_db_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
            )

            document_researcher = Agent(
                role='Lead Document Researcher',
                goal='Answer user questions by searching the technical PDF document.',
                backstory="You are a meticulous researcher who uses the `answer_pdf_question` tool to find information in a technical PDF.",
                tools=[pdf_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
            )

            print("\n--- CrewAI Agent is Ready! ---")
            if IS_TRACING_ENABLED:
                project_name = os.getenv("LANGCHAIN_PROJECT", "CrewAI Project")
                print(f"--- LangSmith Tracing is ENABLED for project: '{project_name}' ---")
            else:
                print("--- LangSmith Tracing is DISABLED ---")

            while True:
                try:
                    user_query = input("\n> ")
                    if user_query.lower() == "exit":
                        break

                    user_task = Task(
                        description=(
                            f"Answer the user's question: '{user_query}'. "
                            "First, analyze the question to determine if it is about company data (employees, sales, products) "
                            "or if it requires searching the technical document. "
                            "Delegate the task to the appropriate specialist: the Database Analyst for company data "
                            "or the Document Researcher for document questions. "
                            "Once the specialist provides the information, formulate a final, user-friendly response."
                        ),
                        expected_output="A clear, concise, and friendly answer to the user's original question, based on the information provided by the specialist agent.",
                    )

                    crew = Crew(
                        agents=[database_analyst, document_researcher],
                        tasks=[user_task],
                        process=Process.hierarchical,
                        manager_llm=llm,
                        verbose=True,
                    )

                    print("\n--- Crew is thinking... ---")
                    result = crew.kickoff()

                    print("\nFinal Answer:")
                    print(result)
                    print("\n" + "="*50)

                except (KeyboardInterrupt, EOFError):
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"\n--- FATAL ERROR connecting to MCP servers: {e}")
        print("--- Please ensure the SQL and RAG tool servers are running. ---")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAgent shut down by user.")
