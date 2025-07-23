import os
import asyncio
from pathlib import Path
from typing import List, Union, Dict, Any

from graphrag.Core.GraphRAG import GraphRAG
from graphrag.Option.Config2 import Config
from libs.retriever.rag_dataset import RAGDataset 

class BaseGraphRAG:
    def __init__(self, model_name, model_url, api_key=None, config_path = './graphrag/Light_RAG.yaml', working_dir=None, dataset_path=None, dataset_name=None , embedding_model=None, max_concurrent=20,rebuild=False):
        assert dataset_name is not None, "dataset_name must be provided"
        assert dataset_path is not None, "dataset_path must be provided"

        config = Config.parse(Path(config_path), working_dir=working_dir, dataset_name=dataset_name)
        config.llm.model = model_name
        config.llm.base_url = model_url
        config.llm.max_concurrent = max_concurrent 
        if api_key != None : 
            config.llm.api_key = api_key
        config.graph.force = rebuild
        config.embedding.model = embedding_model

        self.rag = GraphRAG(config)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        

    def build_graph_sync(self):
        asyncio.run(self.build_graph())

    def query_chunks_sync(self, query, text_only = True, top_k = 5):
        text_return, key_node, node_datas, key_edge, edge_datas = asyncio.run(self.query_chunks(query, text_only = text_only, top_k = top_k))
        return text_return, key_node, node_datas, key_edge, edge_datas

    async def build_graph(self):
        try :
            base_dataset = RAGDataset(
                data_path=self.dataset_path
            )
            corpus = base_dataset.get_corpus()
            await self.rag.insert(corpus)
        except Exception as e:
            print(f"Error building graph: {e}")
            raise

    async def query_chunks(self, question: str, text_only :bool, top_k: int):
        text_return, key_node, node_datas, key_edge, edge_datas = await self.rag._querier.query_chunks_only(question, text_only=text_only, top_k=top_k)
        return text_return, key_node, node_datas, key_edge, edge_datas

    async def query_text_only(self, question: str) -> List[str]:
        result = await self.rag.query(question)

        if isinstance(result, list) and isinstance(result[0], dict):
            if 'content' in result[0]:
                return [r['content'] for r in result]
            elif 'chunk' in result[0]:
                return [r['chunk'] for r in result]
            elif 'text' in result[0]:
                return [r['text'] for r in result]
            else:
                return [str(r) for r in result]  # fallback
        elif isinstance(result, list):
            return [str(r) for r in result]
        else:
            return [str(result)]
