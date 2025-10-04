from typing import Literal, Optional
from agents import User

class ExplicitUser(User):
    def __init__(self, 
        name: str = "ExplicitUser",
        description: Optional[str] = None,
        human_input_mode: Literal["ALWAYS", "NEVER"] = "ALWAYS",
        *kwargs):
        super().__init__(name, description, human_input_mode, *kwargs)
    
    def get_human_input(self, prompt):
        return "exit"