import abc
from typing import Optional


class BaseModel(abc.ABC):
    """Abstract class for all models."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.messages = []
        self.system_prompt = system_prompt
        pass

    @abc.abstractmethod
    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        """Send a message to the assistant and return the assistant's response."""
        pass
