import os
from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import base64
import magic
import openpyxl

from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression like '2 + 2 * 3'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {str(e)}"

# Basic browsing simulation (search + preview)
@tool
def web_browser(query: str) -> str:
    """
    Simulates a web browser by performing a search and returning relevant page content.
    For Wikipedia queries, include 'site:wikipedia.org' in the search.
    For YouTube queries, include 'site:youtube.com' in the search.
    """
    try:
        # Enhance the query for better results
        if "wikipedia" in query.lower():
            query = f"site:wikipedia.org {query}"
        elif "youtube" in query.lower() or "video" in query.lower():
            query = f"site:youtube.com {query}"
            
        tavily = TavilySearchResults(max_results=3)  # Limit to top 3 results
        results = tavily.invoke(query)
        
        # Format the results in a more structured way
        formatted_results = []
        for result in results:
            formatted_results.append(f"Title: {result.get('title', 'No title')}\n"
                                   f"URL: {result.get('url', 'No URL')}\n"
                                   f"Content: {result.get('content', 'No content')}\n")
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
def execute_python(code: str) -> str:
    """Compiles and executes a Python snippet (sandboxed)."""
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "[Executed]"))
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def read_pdf(filename: str) -> str:
    """
    Extracts and returns text from the first few pages of a PDF file.
    """
    if not os.path.exists(filename):
        return f"Error: File '{filename}' not found."
        
    doc = None
    try:
        doc = fitz.open(filename)
        text = ""
        for page in doc[:3]:  # Limit to first 3 pages for speed
            text += page.get_text()
        return text[:1000]  # Limit output to 1000 characters
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    finally:
        if doc:
            doc.close()
    

@tool
def read_spreadsheet(file_name: str) -> str:
    """
    Reads spreadsheet data (CSV, XLSX, XLS, TSV) and returns the first few rows as a formatted string.
    """
    try:
        ext = os.path.splitext(file_name)[-1].lower()
        
        if ext == ".csv":
            df = pd.read_csv(file_name)
        elif ext == ".tsv":
            df = pd.read_csv(file_name, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_name, engine="openpyxl" if ext == ".xlsx" else "xlrd")
        else:
            return f"Unsupported file extension: {ext}"

        return df.head().to_string(index=False)
    
    except Exception as e:
        return f"Error reading spreadsheet: {str(e)}"
    
@tool
def recognize_image(image_description: str) -> str:
    """
    Uses a ChatGPT model to recognize objects in an image based on a textual description.
    Returns the model's response.
    """
    # Initialize ChatGPT model
    chat_model = ChatOpenAI(model="gpt-4")

    try:
        prompt = f"Describe the objects or scene in the image based on the following description: '{image_description}'."
        response = chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error processing image description: {str(e)}"
