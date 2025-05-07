"""
PDF loading and processing module.
"""
import os
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_DIR
from gemini_api import GeminiAPI

class PDFProcessor:
    """Class for loading and processing PDF documents."""

    def __init__(self):
        """Initialize the PDF processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        # Initialize Gemini API for embeddings
        gemini_api = GeminiAPI()
        self.embeddings = gemini_api.get_embeddings()
        self.PDF_DIR = PDF_DIR

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a PDF file and split it into chunks.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of document chunks with text and metadata.
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add source filename to metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def load_pdfs_from_directory(self) -> List[Dict[str, Any]]:
        """
        Load all PDFs from the configured PDF directory.
        Specifically looks for LTE.pdf file for training.

        Returns:
            List of document chunks from all PDFs.
        """
        all_chunks = []

        # Check if directory exists
        if not os.path.exists(PDF_DIR):
            print(f"PDF directory {PDF_DIR} does not exist.")
            return all_chunks

        # Look specifically for LTE.pdf first
        lte_pdf_path = os.path.join(PDF_DIR, "LTE.pdf")
        if os.path.exists(lte_pdf_path):
            try:
                print(f"Found LTE.pdf, processing for training...")
                chunks = self.load_pdf(lte_pdf_path)
                all_chunks.extend(chunks)
                print(f"Processed LTE.pdf: {len(chunks)} chunks extracted")
                # If we found and processed LTE.pdf, return immediately
                return all_chunks
            except Exception as e:
                print(f"Error processing LTE.pdf: {str(e)}")
        else:
            print("LTE.pdf not found in the PDF directory. Please add this file for training.")

            # Fallback: Process other PDF files if LTE.pdf is not available
            print("Processing other available PDF files as fallback...")
            for filename in os.listdir(PDF_DIR):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(PDF_DIR, filename)
                    try:
                        chunks = self.load_pdf(file_path)
                        all_chunks.extend(chunks)
                        print(f"Processed {filename}: {len(chunks)} chunks extracted")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

        return all_chunks
