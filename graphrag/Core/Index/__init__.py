"""RAG factories"""

from graphrag.Core.Index.EmbeddingFactory import get_rag_embedding
from graphrag.Core.Index.IndexFactory import get_index
from graphrag.Core.Index.IndexConfigFactory import get_index_config

__all__ = ["get_rag_embedding", "get_index", "get_index_config"]