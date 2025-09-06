import gradio as gr
import os
import re
import time
import requests
import argparse

from libs.retriever.simple_retriever import Retriever

# os.environ["NO_PROXY"] = "localhost"
UI_PORT = 2222
BACK_END_PORT = 8080

class ChatUI:
    def __init__(self, ui_port=UI_PORT, backend_port=BACK_END_PORT): 
        self.ui_port = ui_port
        self.backend_port = backend_port
        self.chat_history = []

        self.role_name = None
        self.roles_sentences = {}

    def user_submit(self, user_input, chat_history):
        chat_history = chat_history + [{"role": "user", "content": user_input}]
        return chat_history, "", chat_history

    def chat(self, user_input):
        try:
            url = f"http://localhost:{self.backend_port}/chat"
            response = requests.post(
                url,
                json={"message": user_input},
                timeout=60
            )
            print(response)
            if response.status_code != 200:
                return f"[ERROR] Backend returned status code {response.status_code}, {response}"

            data = response.json()

            if "reply" not in data:
                return "[ERROR] Invalid response format: 'reply' field missing."

            return data["reply"]

        except requests.exceptions.RequestException as e:
            return f"[ERROR] Request failed: {e}"
        except ValueError:
            return "[ERROR] Failed to parse response as JSON"


    def model_reply(self, chat_history):
        last_user_msg = chat_history[-1]
        if last_user_msg["role"] != "user":
            # self.logger.warning("The last message is not from the user, skipping model reply.")
            return chat_history, chat_history
        else:
            last_user_msg = last_user_msg["content"]

        response = self.chat(last_user_msg)
        chat_history = chat_history + [{"role": "assistant", "content": response}]
        return chat_history, chat_history

    def fetch_role_state(self):
        try:
            response = requests.get(f"http://localhost:{self.backend_port}/role_state", timeout=10)
            if response.status_code != 200:
                return f"[ERROR] 请求失败: {response.status_code}"

            data = response.json()
            current_role = data.get("current_role", "（未设置）")
            roles = data.get("available_roles", [])

            self.role_name = current_role
            self.roles_sentences = roles

            return gr.update(choices=roles, value=current_role)

        except Exception as e:
            return gr.update(choices=["加载失败"], value="加载失败")

    def switch_role_wrapper(self, role_name):
        try:
            print(f'switch role to : {role_name}')
            response = requests.post(
                f"http://localhost:{self.backend_port}/switch_role",
                json={"roles": role_name},
                timeout=15
            )
            if response.status_code != 200:
                msg = f"[切换失败] {response.text}"
                role_state = self.fetch_role_state()
                return [], [], role_state

            data = response.json()
            self.role_name = data.get("current_role", role_name)
            self.chat_history = []
            role_state = self.fetch_role_state()

            return [], [], role_state

        except Exception as e:
            role_state = self.fetch_role_state()
            return [], [], role_state

    def launch_ui(self):
        with gr.Blocks() as demo:
            '''
            components
            '''
            gr.Markdown("<h1 style='font-size:26px; font-weight:bold;'>Test-Time-Matching Demo</h1>")

            chatbot = gr.Chatbot(label="Roleplay Chat", show_label=True, type="messages")
            state = gr.State(value=[])

            user_input = gr.Textbox(label="输入信息", placeholder="请输入您想和角色交流的内容")
            send_button = gr.Button("发送")
            
            role_selector = gr.Dropdown(
                choices=["loading"], label="切换角色", value="loading"
            )

            '''
            '''
            demo.load(
                fn=self.fetch_role_state,
                inputs=[],
                outputs=[role_selector]
            )

            role_selector.change(
                fn=self.switch_role_wrapper,
                inputs=role_selector,
                outputs=[chatbot, state, role_selector]
            )

            send_button.click(
                fn=self.user_submit,
                inputs=[user_input, state],
                outputs=[chatbot, user_input, state]
            ).then(
                fn=self.model_reply,
                inputs=state,
                outputs=[chatbot, state]
            )

            demo.launch(server_name="localhost", server_port=self.ui_port, inbrowser=False, share=False)

if __name__ == "__main__":
    UI = ChatUI(ui_port=UI_PORT, backend_port=BACK_END_PORT)
    UI.launch_ui()
