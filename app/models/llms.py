import os
from typing import Optional
import typing as tp
import httpx
import json
import re

import requests
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.gigachat import GigaChat
from typing import Optional

from app.models.base import BaseModel


class YandexGPT(BaseModel):
    """See more on https://yandex.cloud/en-ru/docs/foundation-models/concepts/yandexgpt/models"""

    model_urls = {
        "lite": "gpt://{}/yandexgpt-lite/latest",
        "pro": "gpt://{}/yandexgpt/latest",  # Restricted by competition rules, but you can test accuracy with it
    }

    def __init__(
        self,
        token: str,
        folder_id: str,
        model_name: str = "lite",
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 2000,
    ) -> None:
        super().__init__(system_prompt)
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-folder-id": folder_id,
        }
        self.model_url = YandexGPT.model_urls[model_name].format(folder_id)
        self.completion_options = {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        }

    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "text": self.system_prompt})

        self.messages.append({"role": "user", "text": user_message})

        json_request = {
            "modelUri": self.model_url,
            "completionOptions": self.completion_options,
            "messages": self.messages,
        }

        response = requests.post(self.api_url, headers=self.headers, json=json_request)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        response_data = response.json()
        assistant_message = response_data["result"]["alternatives"][0]["message"]["text"]
        self.messages.append({"role": "assistant", "text": assistant_message})
        return assistant_message



class DuckDuckGoLLM(BaseModel):
    def __init__(
        self, 
        system_prompt: Optional[str] = None,

        llm_chat_url: str = "https://duckduckgo.com/duckchat/v1/chat", 
        model: tp.Literal[
            "claude-3-haiku-20240307",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "gpt-4o-mini"
        ] = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        x_vqd_4: str =  "4-248234416339843192477661751153195587515",
    ):
        super().__init__(system_prompt)

        self.url = llm_chat_url
        self.model = model
        self.system_prompt = system_prompt
        
        self.client = httpx.Client(headers={
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9,ru-RU;q=0.8,ru;q=0.7",
            "content-type": "application/json",
            "cookie": "dcm=5",
            "dnt": "1",
            "origin": "https://duckduckgo.com",
            "priority": "u=1, i",
            "referer": "https://duckduckgo.com/",
            "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "x-vqd-4": x_vqd_4,
        })

    def _assemble_request_body(self, messages):
        return {
            "model": self.model,
            "messages": messages
        }

    def __parse_chunk(self, chunk: str) -> str :
        if len(chunk) == 0:
            return ""
        payload = chunk[5:]
        try:
            return json.loads(payload)["message"]
        except Exception as e:
            print(e, chunk)
            return ""
        
    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "text": self.system_prompt})

        response = ""
        
        self.messages.append({"role": "user", "text": user_message})

        with self.client.stream("POST", self.url,  json=self._assemble_request_body(self.messages), timeout=120) as r:
            for chunk in r.iter_lines():
                new_data = self.__parse_chunk(chunk)
                response += new_data

        self.messages.append({"role": "assistant", "text": response})

        return self.messages


class GigaChatWrapper(BaseModel):
    def __init__(
        self, 
        credentials, # Auth token
        model: str = "GigaChat", # ["GigaChat", "GigaChat-Pro"]
        system_prompt: Optional[str] = None,
    ):
        super().__init__(system_prompt)

        self.giga = GigaChat(credentials=credentials, model=model, verify_ssl_certs=False)  

        self.roles = {
            HumanMessage: "system",
            SystemMessage: "system",
            AIMessage: "assistant",
        }

    def _unwrap_giga_messages(self, messages):
        parsed_messages = []

        for message in messages:
            parsed_messages.append({"role": self.roles[type(message)], "text": message.content})

        return parsed_messages
        
    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append(SystemMessage(content=self.system_prompt))
        
        self.messages.append(HumanMessage(content=user_message))

        output = self.giga(self.messages)

        self.messages.append(output)

        # return self._unwrap_giga_messages(self.messages)

        return output.content


    