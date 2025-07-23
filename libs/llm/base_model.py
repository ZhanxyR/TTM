import os
from openai import OpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

class ApiChatModel:
    def __init__(self, model="qwen2.5-7b-instruct", api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.kwargs = kwargs

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        

    def _convert_message(self, msg: BaseMessage):
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        elif isinstance(msg, BaseMessage):
            return {"role": getattr(msg, "role", "user"), "content": msg.content}
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")

    def _convert_input(self, input_data):
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
        elif isinstance(input_data, ChatPromptValue):
            return [self._convert_message(m) for m in input_data.messages]
        elif isinstance(input_data, list):
            return [self._convert_message(m) for m in input_data]
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def invoke(self, messages_or_prompt):
        converted_messages = self._convert_input(messages_or_prompt)

        # for qwen3
        if 'qwen3' in self.model.lower():
            self.kwargs.setdefault("extra_body", {}) 
            self.kwargs["extra_body"].setdefault("chat_template_kwargs", {})
            self.kwargs["extra_body"]["chat_template_kwargs"].setdefault("enable_thinking", False)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            **self.kwargs
        )

        reply = completion.choices[0].message.content

        return AIMessage(content=reply)
