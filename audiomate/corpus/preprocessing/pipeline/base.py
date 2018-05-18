import abc

import networkx as nx


class Step(metaclass=abc.ABCMeta):
    """
    This class is the base class for a step in a processing pipeline.

    It handles the procedure of executing the pipeline. It makes sure the steps are computed in the correct order.
    It also provides the correct inputs to every step.

    Every step has to provide a ``compute`` method which is the actual processing.

    Args:
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, name=None):
        self.graph = nx.DiGraph()
        self.name = name

    def process(self, data, sampling_rate, **kwargs):
        """
        Execute the processing of this step and all dependent parent steps.
        """
        steps = nx.algorithms.dag.topological_sort(self.graph)
        step_results = {}

        for step in steps:
            parent_steps = [edge[0] for edge in self.graph.in_edges(step)]

            if len(parent_steps) == 0:
                res = step.compute(data, sampling_rate, **kwargs)
            elif isinstance(step, Computation):
                parent_output = step_results[parent_steps[0]]
                res = step.compute(parent_output, sampling_rate, **kwargs)
            else:
                # use step.parents to make sure the same order is kept as in the constructor of the reduction
                parent_outputs = [step_results[parent] for parent in step.parents]
                res = step.compute(parent_outputs, sampling_rate, **kwargs)

            if step == self:
                return res
            else:
                step_results[step] = res

    @abc.abstractmethod
    def compute(self, data, sampling_rate, **kwargs):
        pass


class Computation(Step, metaclass=abc.ABCMeta):
    """
    Base class for a computation step.

    Args:
        parent (Step, optional): The parent step this step depends on.
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, parent=None, name=None):
        super(Computation, self).__init__(name=name)

        self.graph.add_node(self)

        if parent is not None:
            self.graph.add_nodes_from(parent.graph.nodes)
            self.graph.add_edges_from(parent.graph.edges)
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Computation'
        else:
            return self.name


class Reduction(Step, metaclass=abc.ABCMeta):
    """
    Base class for a reduction step.

    Args:
        parents (list): List of parent steps this step depends on.
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, parents, name=None):
        super(Reduction, self).__init__(name=name)

        self.parents = list(parents)
        self.graph.add_node(self)

        for index, parent in enumerate(parents):
            self.graph.add_nodes_from(parent.graph.nodes)
            self.graph.add_edges_from(parent.graph.edges)
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Reduction'
        else:
            return self.name
