"""
Google Gemini API integration for the RAG chatbot.
"""
import os
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config import GEMINI_API_KEY, GEMINI_MODEL

class GeminiAPI:
    """Google Gemini API integration."""

    def __init__(self):
        """Initialize the Gemini API."""
        self.api_key_valid = self._check_api_key()

        if self.api_key_valid:
            try:
                # Configure the Gemini API
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Initialize the embeddings model
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GEMINI_API_KEY,
                )
                
                # Initialize the chat model
                self.chat_model = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.2,
                )
                
                print("Gemini API initialized successfully.")
            except Exception as e:
                print(f"Error initializing Gemini API: {str(e)}")
                self.api_key_valid = False
                self.embeddings = None
                self.chat_model = None
        else:
            print("WARNING: No valid Gemini API key found. Limited functionality available.")
            self.embeddings = None
            self.chat_model = None

    def _check_api_key(self) -> bool:
        """Check if the Gemini API key is valid."""
        if not GEMINI_API_KEY or len(GEMINI_API_KEY) < 20:
            print("WARNING: Invalid Gemini API key. Please set a valid key in the .env file.")
            return False
        return True

    def get_embeddings(self) -> Optional[Embeddings]:
        """Get the embeddings model."""
        return self.embeddings if self.api_key_valid else None

    def get_chat_model(self) -> Optional[BaseChatModel]:
        """Get the chat model."""
        return self.chat_model if self.api_key_valid else None

    def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the Gemini model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            The generated response text.
        """
        if not self.api_key_valid or self.chat_model is None:
            return "Error: Gemini API key is invalid or model is not initialized."

        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # Generate response
            response = self.chat_model.invoke(langchain_messages)
            return response.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def test_connection(self) -> bool:
        """Test the connection to the Gemini API."""
        if not self.api_key_valid:
            return False

        try:
            # Simple test query
            response = self.generate_response([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, are you working?"}
            ])
            return "working" in response.lower() or "hello" in response.lower()
        except Exception as e:
            print(f"Error testing Gemini API connection: {str(e)}")
            return False
