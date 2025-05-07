"""
Simple conversation handler for RAG-based chatbot.
"""
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage

from rag_engine import RAGEngine
from prompt_templates import create_rag_prompt, create_chat_prompt

# Initialize the RAG engine
rag_engine = RAGEngine()

# Get LLM from RAG engine
llm = rag_engine.llm if rag_engine.api_key_valid else None

def process_message(message: str, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Process a user message and generate a response.

    Args:
        message: The user's message.
        conversation_history: Optional conversation history.

    Returns:
        A dictionary containing the updated conversation state.
    """
    if conversation_history is None:
        conversation_history = []

    # Add the user message to the history
    user_message = HumanMessage(content=message)
    messages = conversation_history + [user_message]

    # Check if API key is valid
    if not rag_engine.api_key_valid or llm is None:
        error_message = (
            "I'm sorry, but I can't process your request because the Gemini API key is missing or invalid. "
            "Please set a valid API key in the .env file and restart the application. "
        )
        ai_message = AIMessage(content=error_message)
        messages = messages + [ai_message]

        return {
            "messages": messages,
            "context": [],
            "current_question": message,
            "use_rag": False,
            "sources": []
        }

    try:
        # Retrieve relevant context
        relevant_docs = rag_engine.retrieve_relevant_context(message)
        sources = rag_engine.get_sources_from_docs(relevant_docs)

        # Determine if we should use RAG based on whether we found relevant context
        use_rag = len(relevant_docs) > 0

        # Generate response
        if use_rag:
            # Create RAG prompt
            prompt = create_rag_prompt(relevant_docs, message)

            # Generate response
            response = llm.invoke(prompt.to_messages())

            # Add source information to the response
            sources_text = ""
            if sources:
                sources_text = "\n\nSources: " + ", ".join(sources)

            # Create AI message with the response
            ai_message = AIMessage(content=response.content + sources_text)
        else:
            # Create chat prompt for general response
            prompt = create_chat_prompt(message)

            # Generate response
            response = llm.invoke(prompt.to_messages())

            # Create AI message with the response
            ai_message = AIMessage(content=response.content)

        # Update messages
        messages = messages + [ai_message]

        # Return the updated state
        return {
            "messages": messages,
            "context": relevant_docs,
            "current_question": message,
            "use_rag": use_rag,
            "sources": sources
        }
    except Exception as e:
        error_message = f"I'm sorry, but an error occurred while processing your request: {str(e)}"
        ai_message = AIMessage(content=error_message)
        messages = messages + [ai_message]

        return {
            "messages": messages,
            "context": [],
            "current_question": message,
            "use_rag": False,
            "sources": []
        }
