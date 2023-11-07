import openai
import os

engine_list = ['gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k', 'text-embedding-ada-002']

class Config:
    OPENAI_API_TYPE = "azure"
    OPENAI_API_BASE = "your_api_base"
    OPENAI_API_VERSION = "2023-07-01-preview"
    OPENAI_API_KEY = "your_api_key"
    MAX_TURNS = 10
    os.makedirs('logs', exist_ok=True)
    SD_URL = ""
    SD_T2I_API = ""
    serpapi_api_key = ""
    serper_api_key = ""

class OpenAIChat:
    def __init__(self, model_name="gpt-35-turbo"):
        openai.api_type = Config.OPENAI_API_TYPE
        openai.api_base = Config.OPENAI_API_BASE
        openai.api_version = Config.OPENAI_API_VERSION
        openai.api_key = Config.OPENAI_API_KEY
        if model_name not in engine_list:
            raise ValueError("The model name is not in the list of available models among gpt-35-turbo, gpt-35-turbo-16k, gpt-4, gpt-4-32k, text-embedding-ada-002.")
        self.model_name = model_name
        self.messages = [{"role": "system", "content": "You are an experienced python programmer which can write codes to fulfill user's requests."}]

    def chat(self, messages):
        message = ""
        if isinstance(messages, list):
            message += (tmp_message for tmp_message in messages)
        elif isinstance(messages, str):
            message = messages
        else:
            raise TypeError("Messages must be a list or str.")
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            engine=self.model_name,
            messages=self.messages,
            #max_tokens=2048
        )
        self.messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        return response['choices'][0]['message']['content']
    
    def set_system_prompt(self, prompt):
        system_messages = ""
        if isinstance(prompt, list):
            system_messages += (system_message for system_message in prompt)
        elif isinstance(prompt, str):
            system_messages = prompt
        else:
            raise TypeError("System messages must be a list or string.")
        self.messages[0]['content'] = prompt

    def get_history(self):
        return self.messages
    
if __name__ == "__main__":
    openai_chat = OpenAIChat()
    query = ""
    print(openai_chat.chat(query))