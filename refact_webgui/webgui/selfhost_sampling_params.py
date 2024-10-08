import time

from fastapi import Query
from pydantic import BaseModel

from typing import List, Union, Optional, Dict, Any


def clamp(lower, upper, x):
    return max(lower, min(upper, x))


class NlpSamplingParams(BaseModel):
    max_tokens: int = 500
    temperature: float = 0.2
    top_p: float = 1.0
    top_n: int = 0
    stop: Union[List[str], str] = []

    def clamp(self):
        self.temperature = clamp(0, 4, self.temperature)
        self.top_p = clamp(0.0, 1.0, self.top_p)
        self.top_n = clamp(0, 1000, self.top_n)
        self.max_tokens = clamp(0, 8192, self.max_tokens)
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_n": self.top_n,
            "max_tokens": self.max_tokens,
            "created": time.time(),
            "stop_tokens": self.stop,
        }


class NlpCompletion(NlpSamplingParams):
    model: str = Query(pattern="^[a-z/A-Z0-9_\.\-\:]+$")
    prompt: str
    n: int = 1
    echo: bool = False
    stream: bool = False
    mask_emails: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str
    # TODO: validate using pydantic
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatContext(NlpSamplingParams):
    model: str = Query(pattern="^[a-z/A-Z0-9_\.\-]+$")
    messages: List[ChatMessage]
    # TODO: validate using pydantic
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    stream: Optional[bool] = True
    n: int = 1


class EmbeddingsStyleOpenAI(BaseModel):
    input: Union[str, List[str]]
    model: str = Query(pattern="^[a-z/A-Z0-9_\.\-]+$")
