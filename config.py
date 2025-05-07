"""
Configuration settings for the RAG chatbot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set your API key in .env file
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Gemini model to use

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Number of relevant chunks to retrieve

# FastAPI settings
API_TITLE = "LTE Document Chatbot API"
API_DESCRIPTION = "API for a RAG-based chatbot using Google Gemini for LTE documentation"
API_VERSION = "0.1.0"
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
