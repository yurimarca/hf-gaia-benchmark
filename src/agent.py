import os
import wikipedia
import logging

from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import MultiVectorRetriever
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from supabase.client import Client, create_client
from langsmith import Client as LangSmithClient
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler

from tools import calculator, read_pdf, read_spreadsheet, recognize_image, execute_python, wiki_search, arvix_search, web_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize LangSmith
langsmith_client = LangSmithClient()
run_collector = RunCollectorCallbackHandler()
tracer = LangChainTracer(
    project_name="gaia-agent",
    client=langsmith_client
)
callback_manager = CallbackManager([tracer, run_collector])

# System prompt guiding tool use and step-by-step reasoning
system_prompt = SystemMessage(content="""
You are a helpful assistant tasked with answering questions using a set of tools. 
Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: 
FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
Your answer should only start with "FINAL ANSWER: ", then follows with the answer. 
""")

tools = [calculator, 
        read_pdf, 
        read_spreadsheet, 
        recognize_image, 
        execute_python, 
        web_search,
        wiki_search,
        arvix_search]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    callbacks=[tracer]
)
llm_with_tools = llm.bind_tools(tools)

def build_graph():

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        state["messages"] = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(state["messages"])

        return {"messages": [response]}

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile the graph agent
    agent = builder.compile()

    return agent

# test
if __name__ == "__main__":
    question = """What is the name of the country spelled backwards that won 
    FIFA world cup hosted in a South America country that is not from South America?"""
    logger.info(f"Starting agent with question: {question}")
    
    # Build the graph
    graph = build_graph()
    # Run the graph with increased recursion limit
    messages = [HumanMessage(content=question)]
    logger.info("Invoking graph...")
    messages = graph.invoke(
        {"messages": messages},
        config={"recursion_limit": 30}  # Set a reasonable recursion limit
    )
    
    logger.info("Processing final messages:")
    for m in messages["messages"]:
        logger.info(f"Message: {m.content[:200]}...")
        m.pretty_print()
