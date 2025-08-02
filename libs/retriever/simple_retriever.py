import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Retriever:
    def __init__(self, chunks, embedding_model="Qwen/Qwen3-Embedding-0.6B", rerank_model='Qwen/Qwen3-Reranker-0.6B', k=20, cached_dir ="cache/vector_cache", device="cuda", debug=False):

        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(model_name=embedding_model),
            persist_directory=cached_dir,
            client_settings=Settings(anonymized_telemetry=False)
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            chunks, 
            k=k 
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": k}),
                self.bm25_retriever
            ],
            weights=[0.5, 0.5]  
        )
        
        self.debug = debug
        self.device = device
        self.rerank_model = rerank_model

        '''
        switch reranker model
        '''
        try:
            if 'qwen3' in self.rerank_model.lower():
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.rerank_model)
                if self.reranker_tokenizer.pad_token is None:
                    self.reranker_tokenizer.pad_token = self.reranker_tokenizer.eos_token
                self.reranker = AutoModelForSequenceClassification.from_pretrained(
                    self.rerank_model,
                ).to(self.device)
                self.reranker.config.pad_token_id = self.reranker_tokenizer.pad_token_id
            else:
                self.reranker = CrossEncoder(
                    self.rerank_model, 
                    device=self.device
                )
        except:
            raise ValueError(f"Load retriever model failed. Model name: {self.rerank_model}.")


    def get_scores(self, pairs):

        try:
            if 'qwen3' in self.rerank_model.lower():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.reranker(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    scores = probs[:, 1].tolist() 
                    
                return scores
            else:
                return self.reranker.predict(pairs)
        except:
            raise ValueError(f"No implementation for rerank_model in \'get_scores\': {self.rerank_model}.")
            

    def retrieve(self, query, top_k=5):
        docs = self.ensemble_retriever.invoke(query)
        
        pairs = [[query, doc.page_content] for doc in docs]

        scores = self.get_scores(pairs)

        del pairs
        torch.cuda.empty_cache()
        
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        results = [doc for doc, _ in ranked_docs[:top_k]]

        del scores, ranked_docs
        torch.cuda.empty_cache()
        
        return results