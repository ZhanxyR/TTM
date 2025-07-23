import pandas as pd
from torch.utils.data import Dataset

class RAGDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.corpus_path = data_path
        self.dataset = pd.read_json(self.corpus_path , lines=True, orient="records")

    def get_corpus(self):
        corpus = pd.read_json(self.corpus_path, lines=True)
        corpus_list = []
        for i in range(len(corpus)):
            corpus_list.append(
                {
                    "title": corpus.iloc[i]["title"],
                    "content": corpus.iloc[i]["context"],
                    "doc_id": i,
                }
            )
        return corpus_list


