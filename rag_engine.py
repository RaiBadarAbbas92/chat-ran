"""
RAG (Retrieval Augmented Generation) engine implementation.
"""
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from config import (
    VECTOR_STORE_PATH,
    TOP_K_RESULTS
)
from pdf_loader import PDFProcessor
from gemini_api import GeminiAPI

class RAGEngine:
    """RAG engine for document retrieval and answer generation."""

    def __init__(self):
        """Initialize the RAG engine."""
        # Initialize Gemini API
        self.gemini_api = GeminiAPI()
        self.api_key_valid = self.gemini_api.api_key_valid

        if self.api_key_valid:
            try:
                self.embeddings = self.gemini_api.get_embeddings()
                self.llm = self.gemini_api.get_chat_model()
                self.vector_store = None
                self.retriever = None
                self.pdf_processor = PDFProcessor()

                # Load vector store if it exists
                self._load_or_create_vector_store()
            except Exception as e:
                print(f"Error initializing RAG engine: {str(e)}")
                self.api_key_valid = False
                self.vector_store = None
                self.retriever = None
                self.pdf_processor = None
        else:
            print("WARNING: No valid Gemini API key found. Limited functionality available.")
            self.vector_store = None
            self.retriever = None
            self.pdf_processor = None

    def _check_api_key(self):
        """Check if the Gemini API key is valid."""
        return self.gemini_api.api_key_valid

    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        vector_store_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")

        if os.path.exists(vector_store_file):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(VECTOR_STORE_PATH),
                    embeddings=self.embeddings,
                    index_name="index"
                )
                print(f"Loaded existing vector store from {VECTOR_STORE_PATH}")
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
                self.vector_store = None

        # If vector store couldn't be loaded, initialize with a dummy document
        if self.vector_store is None:
            # Create a dummy text to initialize the vector store
            dummy_text = "This is a placeholder document to initialize the vector store."

            # Initialize with the dummy document
            self.vector_store = FAISS.from_texts(
                texts=[dummy_text],
                embedding=self.embeddings,
                metadatas=[{"source": "initialization"}]
            )
            print("Created new vector store with placeholder document")

        # Initialize the retriever
        self._setup_retriever()

    def _setup_retriever(self):
        """Set up the document retriever with contextual compression."""
        if self.vector_store is None:
            print("Warning: Vector store is not initialized. Retriever will not be set up.")
            self.retriever = None
            return

        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )

        # Create an LLM-based compressor for better context extraction
        compressor = LLMChainExtractor.from_llm(self.llm)

        # Create a compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents into the vector store.

        Args:
            documents: List of document chunks to index.
        """
        if not self.api_key_valid:
            print("API key is not valid. Cannot index documents.")
            return

        if not documents:
            print("No documents to index.")
            return

        if self.vector_store is None:
            print("Vector store is not initialized. Cannot index documents.")
            return

        # Check if we only have the initialization document
        has_only_placeholder = False
        if hasattr(self.vector_store, 'docstore') and hasattr(self.vector_store.docstore, '_dict'):
            docs = list(self.vector_store.docstore._dict.values())
            if len(docs) == 1 and docs[0].metadata.get("source") == "initialization":
                has_only_placeholder = True

        try:
            if has_only_placeholder:
                # Create a new vector store with the real documents
                print("Replacing placeholder with actual documents")
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                # Add documents to existing vector store
                self.vector_store.add_documents(documents)

            # Save the updated vector store
            self.vector_store.save_local(folder_path=str(VECTOR_STORE_PATH), index_name="index")
            print(f"Indexed {len(documents)} document chunks")

            # Re-setup the retriever with the updated vector store
            self._setup_retriever()
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")

    def index_pdfs_from_directory(self) -> None:
        """Index all PDFs from the configured directory."""
        if not self.api_key_valid:
            print("API key is not valid. Cannot index PDFs.")
            return

        if self.pdf_processor is None:
            print("PDF processor is not initialized. Cannot index PDFs.")
            return

        try:
            documents = self.pdf_processor.load_pdfs_from_directory()
            self.index_documents(documents)
        except Exception as e:
            print(f"Error indexing PDFs from directory: {str(e)}")

    def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The user's query.

        Returns:
            List of relevant document chunks.
        """
        if not self.api_key_valid:
            print("API key is not valid. Cannot retrieve context.")
            return []

        if not self.retriever:
            print("Retriever not initialized.")
            return []

        try:
            relevant_docs = self.retriever.get_relevant_documents(query)

            # Filter out the initialization document
            relevant_docs = [doc for doc in relevant_docs
                            if not (hasattr(doc, 'metadata') and
                                   doc.metadata.get("source") == "initialization")]

            return relevant_docs
        except Exception as e:
            print(f"Error retrieving relevant context: {str(e)}")
            return []

    def get_sources_from_docs(self, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract source information from retrieved documents.

        Args:
            docs: List of retrieved documents.

        Returns:
            List of source filenames.
        """
        if not docs:
            return []

        sources = []
        try:
            for doc in docs:
                if hasattr(doc, 'metadata') and "source" in doc.metadata and doc.metadata["source"] not in sources:
                    # Skip the initialization placeholder document
                    if doc.metadata["source"] != "initialization":
                        sources.append(doc.metadata["source"])
        except Exception as e:
            print(f"Error extracting sources from documents: {str(e)}")
        return sources
