import google.generativeai as genai
import os
import sys

# Get API key from environment or argument
api_key = os.environ.get("GOOGLE_API_KEY")
if len(sys.argv) > 1:
    api_key = sys.argv[1]

if not api_key:
    print("Error: Please provide an API Key as an argument or set GOOGLE_API_KEY environment variable.")
    print("Usage: python check_models.py YOUR_API_KEY")
    sys.exit(1)

try:
    genai.configure(api_key=api_key)

    print("Listing available models...")
    found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            found = True
    
    if not found:
        print("No models found with 'generateContent' capability.")

except Exception as e:
    print(f"Error listing models: {e}")
