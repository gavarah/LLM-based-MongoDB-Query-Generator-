from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def get_llm(provider, api_key=None, model_name=None, temperature=0):
    """
    Factory function to get the LLM instance based on provider.
    """
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("API Key is required for OpenAI")
        return ChatOpenAI(
            api_key=api_key, 
            model=model_name if model_name else "gpt-4o",
            temperature=temperature
        )
    
    elif provider == "Gemini":
        if not api_key:
            raise ValueError("API Key is required for Gemini")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key, 
            model=model_name if model_name else "gemini-2.5-flash",
            temperature=temperature
        )
    
    elif provider == "Local Llama (Ollama)":
        return ChatOllama(
            model=model_name if model_name else "llama3",
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    