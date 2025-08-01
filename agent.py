# file: agent.py
import os
import asyncio

# Load environment variables from .env file at the very top.
# This ensures all configurations are available before any other modules are loaded.
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- Configuration Checks ---
# Check for mandatory Google API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file.")

# Check for optional LangSmith configuration
IS_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING") == "true" and os.getenv("LANGCHAIN_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def main():
    """
    Sets up and runs the asynchronous agent application using LangGraph's ToolNode.
    """
    print("Setting up tools using MultiServerMCPClient...")
    try:
        client = MultiServerMCPClient(
            {
                "sql_server": {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable_http"},
                "rag_server": {"url": "http://127.0.0.1:8002/mcp", "transport": "streamable_http"},
            }
        )
        tools = await client.get_tools()
        for tool in tools:
            if tool.name == "answer_database_question":
                tool.name = "CompanyDatabaseTool"
            elif tool.name == "answer_pdf_question":
                tool.name = "PDFDocumentSearchTool"

        print("\nSuccessfully loaded and adapted tools:")
        for t in tools:
            print(f"- Tool Name: '{t.name}', Description: {t.description}")

    except Exception as e:
        print(f"\n--- FATAL ERROR during tool setup: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Agent and Graph Setup ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are a helpful assistant. Your job is to answer user questions by using the provided tools."
        "\n\nHere are your tools:"
        "\n- `CompanyDatabaseTool`: Use this for any questions about company data like employees, products, or sales."
        "\n- `PDFDocumentSearchTool`: Use this for specific questions about the technical PDF on drug formulation and wound healing."
        "\n\nYour instructions:"
        "\n1. When the user asks a question, decide which tool is the most appropriate."
        "\n2. Use the user's actual question as the query for the tool. For general summarization, use the user's request as the query."
        "\n3. After the tool returns a result, use that information to give a final, conversational answer to the user."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    llm_with_tools = llm.bind_tools(tools)
    agent_chain = prompt | llm_with_tools

    def agent_node(state: AgentState):
        return {"messages": [agent_chain.invoke({"messages": state["messages"]})]}
        
    workflow = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # --- Main Application Loop ---
    print("\n--- LangGraph Agent is Ready! ---")
    
    # --- KEY CHANGE: Add a clear status message about tracing ---
    if IS_TRACING_ENABLED:
        project_name = os.getenv("LANGCHAIN_PROJECT", "Default Project")
        print(f"--- LangSmith Tracing is ENABLED for project: '{project_name}' ---")
    else:
        print("--- LangSmith Tracing is DISABLED ---")
        print("(To enable, set LANGCHAIN_TRACING_V2='true' and LANGCHAIN_API_KEY in your .env file)")
        
    while True:
        try:
            user_query = await asyncio.to_thread(input, "\n> ")
            if user_query.lower() == "exit":
                break
            
            inputs = {"messages": [HumanMessage(content=user_query)]}
            
            print("\n--- Agent is thinking... ---")
            final_state = await app.ainvoke(inputs, config={"recursion_limit": 10})
            
            final_answer_message = final_state['messages'][-1]

            print("\nFinal Answer:")
            print(final_answer_message.content)
            print("\n" + "="*50)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent shut down by user.")