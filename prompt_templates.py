"""
Prompt templates for the Gemini model.
"""
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Dict, Any

# System prompt for RAG-based question answering
RAG_SYSTEM_PROMPT = """You are an intelligent assistant specialized in LTE (Long-Term Evolution) technology.
Your task is to answer the user's question using ONLY the information from the provided context.
If the context doesn't contain enough information to answer the question fully, acknowledge the limitations and explain what information is missing.
Do not make up information or use your general knowledge to fill in gaps.

Guidelines:
1. Base your answer solely on the provided context
2. If the context is insufficient, say so clearly
3. Be concise but thorough in your response
4. If appropriate, cite the specific source document from the context
5. Format your response in a clear, readable manner using markdown when helpful
6. Focus on providing accurate technical information about LTE technology

Remember, your goal is to provide accurate information from the context, not to demonstrate your general knowledge."""

# Human message template for RAG-based question answering
RAG_HUMAN_TEMPLATE = """
Context information is below:
--------------------
{context}
--------------------

Given the context information and not prior knowledge, answer the question about LTE technology: {question}
"""

def create_rag_prompt(context: List[Dict[str, Any]], question: str) -> ChatPromptTemplate:
    """
    Create a prompt for RAG-based question answering.

    Args:
        context: List of retrieved document chunks.
        question: The user's question.

    Returns:
        A formatted ChatPromptTemplate.
    """
    # Format the context as a string
    context_str = "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}"
                              for i, doc in enumerate(context)])

    # Create the prompt template
    system_message_prompt = SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(RAG_HUMAN_TEMPLATE)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

    return chat_prompt.format_prompt(context=context_str, question=question)

# System prompt for conversational chat (when no context is available)
CHAT_SYSTEM_PROMPT = """You are an LTE technology expert assistant powered by Google Gemini.
Your goal is to provide informative, relevant, and helpful responses to the user's questions about LTE technology.
Always prioritize accuracy and clarity in your responses.
If you're unsure about something, acknowledge your uncertainty rather than making up information.
Be concise but thorough in your explanations.
If the user asks about topics unrelated to LTE, politely redirect them to ask about LTE technology."""

# Human message template for conversational chat
CHAT_HUMAN_TEMPLATE = """{input}"""

def create_chat_prompt(input_text: str) -> ChatPromptTemplate:
    """
    Create a prompt for general conversational chat.

    Args:
        input_text: The user's input.

    Returns:
        A formatted ChatPromptTemplate.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(CHAT_SYSTEM_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(CHAT_HUMAN_TEMPLATE)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

    return chat_prompt.format_prompt(input=input_text)
