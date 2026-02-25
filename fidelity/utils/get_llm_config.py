def get_llm_config(
    client: str = "maritaca",
    temperature: float = 0.0,
    model: str = "sabia-3.1",
    api_key: str = None,
    base_url: str = None,
    response_format=None
):
    config = {
        "client": client,
        "temperature": temperature,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }

    if response_format == "json_object":
        config["response_format"] = {"type": "json_object"}

    return {
        "config_list": [config]
    }
