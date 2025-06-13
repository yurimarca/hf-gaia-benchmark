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

from tools import calculator, read_pdf, read_spreadsheet, recognize_image, execute_python, web_browser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# System prompt guiding tool use and step-by-step reasoning
system_prompt = SystemMessage(content="""
You are a helpful assistant tasked with answering questions using a set of tools. 
Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: 
FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
Your answer should only start with "FINAL ANSWER: ", then follows with the answer. 
""")

# Initialize LLM
tools = [calculator, 
        read_pdf, 
        read_spreadsheet, 
        recognize_image, 
        execute_python, 
        web_browser]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Setup Supabase vector store
supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_SERVICE_KEY"]

vector_store = SupabaseVectorStore(
    client=create_client(supabase_url, supabase_key),
    embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    table_name="documents",
    query_name="match_documents",
)

retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

def build_graph():
    
    def retriever_node(state: MessagesState) -> dict:
        try:
            logger.info(f"Retriever node processing message: {state['messages'][0].content[:100]}...")
            similar_questions = vector_store.similarity_search(
                state["messages"][0].content,
                k=1  # Get only the most similar question
            )
            if similar_questions:
                logger.info(f"Found similar question: {similar_questions[0].page_content[:100]}...")
                example_msg = HumanMessage(
                    content=f"Here I provide a similar question and answer for reference: \n\n{similar_questions[0].page_content}",
                )
                return {"messages": [system_prompt] + state["messages"] + [example_msg]}
            else:
                logger.info("No similar questions found")
                return {"messages": [system_prompt] + state["messages"]}
        except Exception as e:
            logger.error(f"Error in retriever node: {e}", exc_info=True)
            return {"messages": [system_prompt] + state["messages"]}

    def assistant_node(state: MessagesState) -> dict:
        try:
            logger.info("Assistant node processing messages")
            response = llm_with_tools.invoke(state["messages"])
            logger.info(f"Assistant generated response: {response.content[:200]}...")
            
            # Check if the response contains a final answer
            if "FINAL ANSWER:" in response.content:
                logger.info("Found final answer, stopping recursion")
                return {"messages": [response]}
            else:
                logger.info("No final answer found, continuing conversation")
                return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in assistant node: {e}", exc_info=True)
            raise

    logger.info("Building graph...")
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever_node)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    logger.info("Graph built successfully")
    return builder.compile()

# test
if __name__ == "__main__":
    question = "When did Pele score his first goal in the World Cup?"
    logger.info(f"Starting agent with question: {question}")
    
    # Build the graph
    graph = build_graph()
    # Run the graph with increased recursion limit
    messages = [HumanMessage(content=question)]
    logger.info("Invoking graph...")
    messages = graph.invoke(
        {"messages": messages},
        config={"recursion_limit": 10}  # Set a reasonable recursion limit
    )
    
    logger.info("Processing final messages:")
    for m in messages["messages"]:
        logger.info(f"Message: {m.content[:200]}...")
        m.pretty_print()
