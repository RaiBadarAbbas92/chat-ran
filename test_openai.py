"""
Test script to verify OpenAI API key is working.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key.startswith("sk-your-actual"):
    print("ERROR: Please set a valid OpenAI API key in the .env file")
    print("You can get an API key from https://platform.openai.com/api-keys")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

try:
    # Test API key with a simple completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
    )
    
    # Print the response
    print("API Key is working! Response:")
    print(response.choices[0].message.content)
    print("\nYou can now run the main application with: python main.py")
    
except Exception as e:
    print(f"ERROR: Failed to connect to OpenAI API: {str(e)}")
    print("Please check your API key and internet connection.")
