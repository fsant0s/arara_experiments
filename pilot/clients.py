import os

# Load environment variables from a .env file if present (project root or parent)
def _load_env_from_dotenv():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '.env'),
            os.path.abspath(os.path.join(here, '..', '.env')),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                break
    except Exception:
        # best effort; ignore parsing errors
        pass

_load_env_from_dotenv()

ollama_llama32 = {
    "client": "ollama",  # explicitly indicate default client type
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

gpt_4o= {
    "config_list": [
        {
            "client": "openai",
            "temperature": 0.0,
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
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

openrouter_llama370b = {
    "config_list": [
        {
            "client": "openrouter",
            "temperature": 0.0,
            "model": "google/gemini-pro-1.5",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPEN_ROUTER_API_KEY")
        }
    ]
}

openrouter_gpt4o = {
    "config_list": [
        {
            "client": "openrouter",
            "temperature": 0.0,
            "model": "openai/gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPEN_ROUTER_API_KEY")
        }
    ]
}

openrouter_claude35 = {
    "config_list": [
        {
            "client": "openrouter",
            "temperature": 0.0,
            "model": "anthropic/claude-3.5-sonnet",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPEN_ROUTER_API_KEY")
        }
    ]
}


openrouter_deepseek_chat = {
    "config_list": [
        {
            "client": "openrouter",
            "temperature": 0.0,
            "model": "deepseek/deepseek-chat",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPEN_ROUTER_API_KEY")
        }
    ]
}