import faiss
import os
from graphrag.Core.Common.BaseFactory import ConfigBasedFactory
# from Core.Index.ColBertIndex import ColBertIndex
from graphrag.Core.Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    # ColBertIndexConfig,
    FAISSIndexConfig
)
from graphrag.Core.Index.VectorIndex import VectorIndex
from graphrag.Core.Index.FaissIndex import FaissIndex
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage


class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            VectorIndexConfig: self._create_vector_index,
            # ColBertIndexConfig: self._create_colbert,
            FAISSIndexConfig: self._create_faiss,

        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig):
        """Key is IndexType."""
        return super().get_instance(config)

    @classmethod
    def _create_vector_index(cls, config):
        return VectorIndex(config)

    # @classmethod
    # def _create_colbert(cls, config: ColBertIndexConfig):
    #     return ColBertIndex(config)

    
    def _create_faiss(self, config):
       return FaissIndex(config)


get_index = RAGIndexFactory().get_index
