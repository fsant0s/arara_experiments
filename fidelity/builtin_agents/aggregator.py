from agents import Agent, Module
from typing import List, Literal, Optional, Union

from agent_messages import TextMessage
from agents.types import Response

class Aggregator(Agent):
    
    def __init__(
        self,
        name: str = "user",
        description: Optional[str] = None,
        human_input_mode: Literal["ALWAYS", "NEVER"] = "ALWAYS",
        **kwargs,
    ):
        if (
            kwargs.get("llm_config")
            and isinstance(kwargs["llm_config"], dict)
            and (kwargs["llm_config"].get("functions") or kwargs["llm_config"].get("tools"))
        ):
            raise ValueError(
                "Aggregator is not allowed to make function/tool calls. Please remove the 'functions' or 'tools' config in 'llm_config' you passed in."
            )

        super().__init__(
            name=name,
            system_message=None,
            **kwargs,
        )
    
        self.unregister_reply_func(Agent._generate_oai_reply)
        self.register_reply([Agent, None], Aggregator.aggregate_items )

    def aggregate_items(
        self,
        sender: Agent,
        messages: Optional[List[dict]] = None,
        config: Optional[Module] = None,
    ) -> Union[Agent, str, None]:
        """Aggregate items from multiple agents."""
        if messages is None:
            messages = self._oai_messages[sender]
       
        # Here you can implement your aggregation logic
        aggregated_content = ""
        for msg in messages:
            aggregated_content += f"{msg['content']}"

        response = Response(
            chat_message=TextMessage(
                content=aggregated_content,
                sender=sender,
                receiver=self,
            )
        )
        
        yield [(True, response)]