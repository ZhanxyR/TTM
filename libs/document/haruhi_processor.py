# import json
import os
import json_repair

from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig


# https://huggingface.co/silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18
class HaruhiProcessor:
    def __init__(self, model='silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18', progress=None, logger=None, debug=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model, trust_remote_code=True,top_k = 2, top_p = 0.9, max_new_tokens = 1000) 

        self.progress = progress
        self.logger = logger
        self.debug = debug

        self.sys_prompt = "给定input paragraph，抽取其中的对话，并输出为json格式 Let's think it step by step \
                            1. summarize input paragraph into bullet format，存储在summary字段 \
                            2. 抽取每一句对话的内容 dialogue，判断每一句话的说话人 said by, 存储在conversations中\n"

    def get_token_len(self, text):
        return len(self.tokenizer.encode(text))

    def get_chunks(self, input_dir, chunk_size=600):

        files = [f for f in os.listdir(input_dir) if f.endswith(('.txt'))]

        chunks = []

        for file in files:
            with open(os.path.join(text, file), 'r', encoding='utf-8') as f:
                text = f.read()

            lines = text.split("\n")

            current_chunk = ""
            current_len = 0

            for line in lines:
                if line.strip() == "":
                    continue
                line_len = self.get_token_len(line)
                if current_len + line_len > chunk_size:
                    if current_len > 0:
                        chunks.append(current_chunk)
                        current_chunk = line
                        current_len = line_len
                else:
                    current_chunk += line+ "\n"
                    current_len += line_len

            if current_len > 0:
                chunks.append(current_chunk)

        return chunks

    def extract_dialogues(self, chunks=None, input_dir=None, chunk_size=600):

        if chunks is not None:
            chunks = [chunk['context'] for chunk in chunks]
        elif input_dir is not None:
            chunks = self.get_chunks(input_dir, chunk_size=chunk_size)
        else:
            raise ValueError("Either chunks or input_dir should be provided")

        model = self.model.eval()

        # dsp_text = ""
        summary = ""
        dialogues = []

        task = self.progress.add_task(description=f"Extracting dialogues from {len(chunks)} chunks", total=len(chunks))

        for i, text in enumerate(chunks):
            if summary == "":
                input = text
            else:
                input = summary + "\n" + text

            response_str, history = model.chat(self.tokenizer, input, history=[], system=self.sys_prompt)

            summary = ""

            try:
                # response = json.loads(response_str)
                response = json_repair.loads(response_str)
                
                summary = response["summary"]

                # dsp_text = dsp_text + "\n" + summary

                conversations = response["conversations"]

                for conversation in conversations:
                    speaker = conversation["said_by"]
                    content = conversation["dialogue"]
                    # dsp_text = dsp_text + "\n" + speaker + " : " + content

                    conversation["chunk_id"] = i

                dialogues.append(conversations)

            except:
                # print(response_str)
                pass

            self.progress.advance(task)

        if not self.debug:
            self.progress.update(task, visible=False)
            self.progress.refresh()

        return dialogues
    
    def haruhi_to_dict(self, haruhi_data):
        dialogues = defaultdict(set)

        for data in haruhi_data:
            for d in data:
                dialogues[d['said_by']].add(d['dialogue'])

        dialogues = {k: list(v) for k, v in dialogues.items()}

        return dialogues
