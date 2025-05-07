"""
Test script to verify Gemini API key is working.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or len(api_key) < 20:
    print("ERROR: Please set a valid Gemini API key in the .env file")
    exit(1)

try:
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Test API key with a simple completion
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, are you working?")
    
    # Print the response
    print("API Key is working! Response:")
    print(response.text)
    print("\nYou can now run the main application with: python main.py")
    
except Exception as e:
    print(f"ERROR: Failed to connect to Gemini API: {str(e)}")
    print("Please check your API key and internet connection.")
