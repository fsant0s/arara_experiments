from typing import Union

from agents import Agent, Module
from .aggregator import Aggregator

from ioflow import IOStream
from formatting_utils import colored

iostream = IOStream.get_default()

import json
import re

def parse_recommendations(raw_content):
    # Se já veio como lista/dict (dependendo do client), só retorna
    if isinstance(raw_content, (list, dict)):
        return raw_content

    if not isinstance(raw_content, str):
        raise TypeError(f"Unsupported content type: {type(raw_content)}")

    text = raw_content.strip()

    # Se veio com bloco ```json ... ``` ou ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Agora 'text' deve ser um JSON puro
    parsed = json.loads(text)

    # Normalizar para lista de dicts
    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        raise ValueError("Parsed content is not a list or dict")

    return parsed


def parallel(
    last_speaker: Agent, module: Module, selector: Agent = None
) -> Union[Agent, str, None]:

    iostream.print(colored("\nStarted executing the recommendation systems...", "yellow"), flush=True)
    aggregator = Aggregator(
        name="aggregator",
    )

    message = {}
    for agent in module.agents:
        for reply in agent.generate_reply(sender=selector):
            agent.send(reply, aggregator, silent=False, request_reply=False)
            raw = reply.chat_message.content
            try:
                recs = parse_recommendations(raw)
                message[agent.name] = recs
            except Exception as e:
                # Log seguro no dicionário de resultados
                message[agent.name] = {
                    "error": f"Failed to parse output: {str(e)}",
                    "raw_output": raw,
        }

    aggregator.send(json.dumps(message, ensure_ascii=False), selector, silent=True, request_reply=False)

    if aggregator not in module.agents: module.agents.append(aggregator)
    
    iostream.print(colored("Finished executing the recommendation systems.\n", "yellow"), flush=True)
    return aggregator