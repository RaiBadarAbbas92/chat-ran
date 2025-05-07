"""
FastAPI application for the RAG chatbot.
"""
import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import API_TITLE, API_DESCRIPTION, API_VERSION, PORT, HOST
from rag_engine import RAGEngine
from conversation_graph import process_message
from langchain_core.messages import HumanMessage, AIMessage

# Initialize the RAG engine
rag_engine = RAGEngine()

# Create the FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class MessageRequest(BaseModel):
    """Request model for chat messages."""
    message: str
    conversation_history: Optional[List[Dict[str, Any]]] = None

class MessageResponse(BaseModel):
    """Response model for chat messages."""
    response: str
    sources: List[str] = []

class IndexResponse(BaseModel):
    """Response model for indexing operations."""
    message: str
    documents_indexed: int

# Define the API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    api_status = "API key is valid" if rag_engine.api_key_valid else "API key is missing or invalid"
    return {
        "message": "LTE Document Chatbot API is running",
        "api_status": api_status,
        "setup_instructions": "Set a valid Gemini API key in the .env file and restart the application if needed."
    }

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """
    Chat endpoint.

    Args:
        request: The chat request containing the user message and conversation history.

    Returns:
        The assistant's response.
    """
    # Convert the conversation history to the correct format
    conversation_history = []
    if request.conversation_history:
        for msg in request.conversation_history:
            if msg["role"] == "user":
                conversation_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                conversation_history.append(AIMessage(content=msg["content"]))

    # Process the message
    result = process_message(request.message, conversation_history)

    # Extract the response and sources
    messages = result["messages"]
    last_message = messages[-1]
    response = last_message.content
    sources = result["sources"]

    return MessageResponse(response=response, sources=sources)

@app.post("/upload-pdf", response_model=IndexResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and index a PDF file.

    Args:
        file: The PDF file to upload and index.

    Returns:
        A message indicating the result of the operation.
    """
    # Check if API key is valid
    if not rag_engine.api_key_valid:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key is missing or invalid. Please set a valid key in the .env file and restart the application."
        )

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # Load and index the PDF
        documents = rag_engine.pdf_processor.load_pdf(temp_file_path)

        # Save the PDF to the PDF directory
        pdf_path = os.path.join(rag_engine.pdf_processor.PDF_DIR, file.filename)
        with open(pdf_path, 'wb') as pdf_file:
            # Reset the file position to the beginning
            await file.seek(0)
            pdf_file.write(await file.read())

        # Index the documents
        rag_engine.index_documents(documents)

        return IndexResponse(
            message=f"Successfully indexed {file.filename}",
            documents_indexed=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@app.post("/index-all-pdfs", response_model=IndexResponse)
async def index_all_pdfs():
    """
    Index all PDFs in the configured directory.

    Returns:
        A message indicating the result of the operation.
    """
    # Check if API key is valid
    if not rag_engine.api_key_valid:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key is missing or invalid. Please set a valid key in the .env file and restart the application."
        )

    try:
        # Count the documents before indexing
        if not hasattr(rag_engine.vector_store, 'docstore') or not hasattr(rag_engine.vector_store.docstore, '_dict'):
            raise HTTPException(status_code=500, detail="Vector store not properly initialized")

        initial_doc_count = len(rag_engine.vector_store.docstore._dict)

        # Index all PDFs
        rag_engine.index_pdfs_from_directory()

        # Count the documents after indexing
        final_doc_count = len(rag_engine.vector_store.docstore._dict)
        documents_indexed = final_doc_count - initial_doc_count

        return IndexResponse(
            message="Successfully indexed all PDFs",
            documents_indexed=documents_indexed
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing PDFs: {str(e)}")

@app.post("/train-with-lte-pdf", response_model=IndexResponse)
async def train_with_lte_pdf():
    """
    Specifically train the model with LTE.pdf.

    Returns:
        A message indicating the result of the operation.
    """
    # Check if API key is valid
    if not rag_engine.api_key_valid:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key is missing or invalid. Please set a valid key in the .env file and restart the application."
        )

    try:
        # Check if LTE.pdf exists
        if not hasattr(rag_engine, 'pdf_processor') or rag_engine.pdf_processor is None:
            raise HTTPException(
                status_code=500,
                detail="PDF processor not initialized. Please check your Gemini API key."
            )

        lte_pdf_path = os.path.join(rag_engine.pdf_processor.PDF_DIR, "LTE.pdf")
        if not os.path.exists(lte_pdf_path):
            raise HTTPException(
                status_code=404,
                detail="LTE.pdf not found. Please upload it first using the /upload-pdf endpoint."
            )

        # Count the documents before indexing
        if not hasattr(rag_engine.vector_store, 'docstore') or not hasattr(rag_engine.vector_store.docstore, '_dict'):
            raise HTTPException(status_code=500, detail="Vector store not properly initialized")

        initial_doc_count = len(rag_engine.vector_store.docstore._dict)

        # Load and index LTE.pdf
        documents = rag_engine.pdf_processor.load_pdf(lte_pdf_path)
        rag_engine.index_documents(documents)

        # Count the documents after indexing
        final_doc_count = len(rag_engine.vector_store.docstore._dict)
        documents_indexed = final_doc_count - initial_doc_count

        return IndexResponse(
            message="Successfully trained with LTE.pdf",
            documents_indexed=documents_indexed
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training with LTE.pdf: {str(e)}")

def main():
    """Run the FastAPI application."""
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

if __name__ == "__main__":
    main()
