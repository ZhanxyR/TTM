"""
Graph Factory.
"""
from graphrag.Core.Graph.BaseGraph import BaseGraph
from graphrag.Core.Graph.ERGraph import ERGraph
from graphrag.Core.Graph.PassageGraph import PassageGraph
from graphrag.Core.Graph.TreeGraph import TreeGraph
from graphrag.Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from graphrag.Core.Graph.RKGraph import RKGraph



class GraphFactory():
    def __init__(self):
        self.creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._crease_passage_graph
        }


    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        return self.creators[config.graph.graph_type](config, **kwargs)

    @staticmethod
    def _create_er_graph(config, **kwargs):
        return ERGraph(
            config.graph, **kwargs
        )

    @staticmethod
    def _create_rkg_graph(config, **kwargs):
        return RKGraph(config.graph, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _create_tree_graph_balanced(config, **kwargs):
        return TreeGraphBalanced(config, **kwargs)

    @staticmethod
    def _crease_passage_graph(config, **kwargs):
        return PassageGraph(config.graph, **kwargs)


get_graph = GraphFactory().get_graph
