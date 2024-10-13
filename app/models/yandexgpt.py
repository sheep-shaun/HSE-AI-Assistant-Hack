import os
from typing import Optional

import requests

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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    yandex_gpt = YandexGPT(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        system_prompt="Ты - профессиональный биолог. Отвечай коротко и по делу в научных терминах.",
    )
    print(yandex_gpt.ask("Кто такой манул?"))
