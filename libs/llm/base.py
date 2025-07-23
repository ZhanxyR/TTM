from langchain_openai import ChatOpenAI

from libs.utils.common import get_default_rich_progress
from libs.utils.logger import get_logger
from libs.llm.prompt import PromptWrapper
from libs.llm.generation_agent import GenerationAgent
from libs.llm.chat_agent import ChatAgent
from libs.llm.base_model import ApiChatModel

class LLM:
    def __init__(self, language='zh', workers=8, logger=None, id=0, progress=None, serial=False, debug=False):
        self.model_name = None
        self.model_url = None
        self.api_key = None
        self.model = None
        self.language = language
        self.workers = workers
        self.logger = logger if logger is not None else get_logger()
        self.id = id

        if progress is None:
            progress = get_default_rich_progress()
            progress.start()
        self.progress = progress

        self.prompt_wrapper = PromptWrapper(language=self.language, logger=self.logger)
        self.generation_agent = None
        self.chat_agent = None

        self.use_graphrag = False
        self.graphrag = None

        self.serial = serial
        self.debug = debug

    @staticmethod
    def set_seed(seed=2025):
        import random
        random.seed(seed)
        import torch
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)

    def create_from_url(self, model_id, api_key, base_url, **kwargs):
        self.model_name = model_id
        self.model_url = base_url
        self.api_key = api_key

        self.model = ApiChatModel(model=model_id, api_key=api_key, base_url=base_url, **kwargs)

    def build_graphrag(self, dataset_path, working_dir ,dataset_name, rebuild, **kwargs):
        self.use_graphrag = True
        
        from libs.retriever.graph_rag import BaseGraphRAG
        self.graphrag = BaseGraphRAG(model_name=self.model_name, model_url=self.model_url, api_key=self.api_key, dataset_path=dataset_path, working_dir=working_dir, dataset_name=dataset_name, rebuild=rebuild, **kwargs)
        self.graphrag.build_graph_sync()

    def switch_language(self, language):
        self.language = language
        self.prompt_wrapper.switch(language)

    def init_role_generation(self):
        self.generation_agent = GenerationAgent(self.model, self.prompt_wrapper, language=self.language, workers=self.workers, logger=self.logger, progress=self.progress, debug=self.debug)

    def init_role_playing(self, role_name, linguistic_retriever, processor, personality='', background='', linguistic_style='', **kwargs):
        self.chat_agent = ChatAgent(self.model, linguistic_retriever, processor, self.prompt_wrapper, language=self.language, workers=self.workers, logger=self.logger, progress=self.progress, id=self.id, use_graphrag=self.use_graphrag, debug=self.debug)
        self.chat_agent.init_role_playing(**kwargs)
        self.chat_agent.init_role(role_name=role_name, personality=personality, background=background, linguistic_style=linguistic_style)

        if self.use_graphrag:
            if self.graphrag is None:
                raise ValueError("GraphRAG is not initialized, please call \'build_graphrag()\' first.")
            self.chat_agent.set_graphrag(self.graphrag)

    def chat(self, message):
        return self.chat_agent.chat(message)

    # def TTM_rewriting(self, query, message, type, k, disable_action, use_graphrag):
    #     return self.chat_agent.test_time_matching(query, message, type=type, k=k, disable_action=disable_action, use_graphrag=use_graphrag)

    def summarize_from_chunks(self, chunks, skip=False):
        return self.generation_agent.summarize_from_chunks_wrapper(chunks, skip, self.serial)
    
    def extract_dialogues(self, chunks):
        return self.generation_agent.extract_dialogues_wrapper(chunks, self.serial)

    def detect_role_entities(self, candidates):
        return self.generation_agent.detect_role_entities_wrapper(candidates, self.serial)
    
    def combine_roles_from_entities(self, entities, max_name_length=10):
        return self.generation_agent.combine_roles_from_entities(entities, max_name_length=max_name_length)
    
    def get_related_chunks(self, chunks, roles):
        return self.generation_agent.get_related_chunks_wrapper(chunks, roles, self.serial)

    def analyze_personality_from_chunks(self, chunks, roles):
        return self.generation_agent.analyze_personality_from_chunks_wrapper(chunks, roles, self.serial)
    
    def extract_background_from_chunks(self, chunks, roles, freq=5):
        return self.generation_agent.extract_background_from_chunks(chunks, roles, freq=freq)
        
    def analyze_linguistic_style_from_sentences(self, sentences, max_words=10):
        return self.generation_agent.analyze_linguistic_style_from_sentences(sentences, max_words=max_words)

