# pylint: disable=invalid-name
"""Namespace for building operators."""
from __future__ import absolute_import as _abs

from tvm.contrib import graph_runtime
from ..compiler import graph_attr, graph_util
from ..compiler.build_module import _update_shape_dtype, _remove_noref_params, build

from .. import graph as _graph
from .. import symbol as sym


def simplify_graph(graph, shape, params):
    """Simplify the graph by applying some optimizations

    Parameters
    ----------
    graph : Graph
        The computation graph

    params: dict
        The original parameters

    shape: dict
        Known shapes

    Returns
    -------
    opt_graph : Graph
        Simplified graph

    opt_params : Graph
        Simplified parameters
    """
    graph = (graph if isinstance(graph, _graph.Graph)
             else _graph.create(graph))
    shape, dtype = _update_shape_dtype(shape, {}, params)
    graph = graph_attr.set_shape_inputs(graph, shape)
    # Simplify inference
    graph = graph.apply(["InferShape", "SimplifyInference"])
    # Fold scale axis
    graph = graph_attr.set_shape_inputs(graph, shape)
    graph = graph.apply(["InferShape", "FoldScaleAxis"])
    # Remove nonref_parameters
    params = _remove_noref_params(params, graph)
    return graph, params


class Quantizer(object):
    def __init__(graph, shape, params):
        """Quantize tool, used to create several views of parameters.

        Parameters
        ----------
        graph : Graph
          The original graph

        shape : dict
          The input shape hint(include batch)

        params : dict
          The original parameters
        """
        graph, params = simplify_graph(graph, shape, params)
        self.shape_dict = shape
        self.orig_graph = graph
        self.orig_params = params
        items = buld(graph, shape=self.shape_dict, params=self.orig_params)
        self.orig_rt = graph_runtime.create(items[0], items[1])
        self.orig_rt.set_input(**items[2])
