from collections import defaultdict
from itertools import count
from .inference import forward
from .graphunit import Node


def vector_jacobian(fun, x):
    """
    :param fun: ufunc without specific x_value like tanh, not tanh(x)
    :param x: specific x_value
    :return: vector_jacobian product of fun
    """
    # start_node = Node(None,lambda x: x, (), {}, [],[])
    start_node = Node.new_root()
    end_value, end_node = forward(start_node, fun, x)
    return lambda g: backward(g, end_node), end_value


def backward(g, end_node):
    """
    Traverse computation graph backwards in topological order from the end node.
    For each node, compute local gradient contribution and accumulate.
    :param g: error/gradient backpropagate from next level
    :param end_node: the last node of graph
    :return: final vector-Jacobian product (gradient) of graph.
    """
    # Save vector-Jacobian product (gradient) for upstream nodes.
    node_grads = {end_node: g}
    # Sum gradient from downstream with all others

    for node in toposort(end_node):
        # print(node.recipe[0].__doc__)
        downstream_grad = node_grads.pop(node)

        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            vjp = primitive_vjps[fun][argnum]
            parent_grad = vjp(downstream_grad, value, *args)
            node_grads[parent] = node_grads.get(parent, 0) + parent_grad
    return downstream_grad


primitive_vjps = defaultdict(dict)


def defvjp(fun, *vjps):
    """
    Register vector-Jacobian product for functions
    :param fun: fun(x, y, ...) = ans
    :param argnum: diff wrt i th variable
    :param vjps:
        vjp_x(g, ans, x, y, ...) = g * df/dx
        vjp_y(g, ans, x, y, ...) = g * df/dy
    :param kwargs: optional
    """
    for argnum, vjp in zip(count(), vjps):
        primitive_vjps[fun][argnum] = vjp


def toposort(end_node):
    """
    topology sort
    generate nodes from very end
    """
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)
    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1
