import os

ollama_llama32 = {
    "config_list": [
        {
            "client": "ollama",
            "temperature": 0.7,
            "model": "llama3.2:latest",
            "base_url": "http://localhost:11434",
        }
    ]
}

groq_llama3370b = {
    "config_list": [
        {
            "client": "groq",
            "temperature": 0.0,
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY")
        }
    ]
}

sabia_31 = {
    "config_list": [
        {
            "client": "maritaca",
            "temperature": 0.0,
            "model": "sabia-3.1",
            "api_key": os.getenv("MARITACA_API_KEY"),
            "base_url":"https://chat.maritaca.ai/api", 
        }
    ]
}

gpt_41 = {
    "config_list": [
        {
            "client": "openai",
            "temperature": 0.0,
            "model": "gpt-4.1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]
}