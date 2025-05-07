# LTE Document Chatbot with Google Gemini

A Retrieval Augmented Generation (RAG) chatbot built with FastAPI, FAISS vector store, and Google's Gemini models, specifically designed for LTE documentation.

## Features

- PDF document ingestion and indexing
- Vector storage using FAISS
- Simple conversation flow management
- FastAPI endpoints for chat and document upload
- Google Gemini integration with optimized prompt templates
- Specialized for LTE (Long-Term Evolution) technical documentation

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The Gemini API key is already set in the `.env` file:
   ```
   # The .env file already contains the Gemini API key
   GEMINI_API_KEY=AIzaSyBtCOBuDYTJ4piuB6r7op5ulRafxxcX7B8
   ```

3. Test your Gemini API key:
   ```bash
   python test_gemini.py
   ```

   If successful, you'll see a confirmation message.

4. Run the application:
   ```bash
   python main.py
   ```

## API Endpoints

- `GET /`: Check if the API is running
- `POST /chat`: Send a message to the chatbot
- `POST /upload-pdf`: Upload and index a PDF file
- `POST /index-all-pdfs`: Index all PDFs in the data/pdfs directory
- `POST /train-with-lte-pdf`: Specifically train the model with LTE.pdf

## Usage

1. Place your LTE.pdf file in the `data/pdfs` directory
2. Train the model with LTE.pdf using the `/train-with-lte-pdf` endpoint
3. Alternatively, upload PDF documents using the `/upload-pdf` endpoint
4. Start chatting with the bot using the `/chat` endpoint
5. The bot will retrieve relevant information from your documents to answer questions about LTE technology using Google's Gemini model

## Project Structure

- `main.py`: FastAPI application and endpoints
- `rag_engine.py`: RAG implementation with vector store
- `pdf_loader.py`: PDF loading and processing
- `conversation_graph.py`: Conversation flow implementation
- `prompt_templates.py`: Prompt templates for Gemini
- `config.py`: Configuration settings
- `gemini_api.py`: Google Gemini API integration
- `test_gemini.py`: Script to test Gemini API connection