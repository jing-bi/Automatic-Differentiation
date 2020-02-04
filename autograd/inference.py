from contextlib import contextmanager
from .graphunit import Container, Node


def forward(start_node, fun, x):
    with trace_stack.new_trace() as trace_id:
        # Wrap 'x_value' in a container.
        start_container = Container.new_container(x, trace_id, start_node)
        end_container = fun(start_container)
        if type(
                end_container
        ) in Container.types and end_container._trace_id == start_container._trace_id:
            return end_container._value, end_container._node
        else:
            return end_container, None


def func_wrapper(f_raw):
    """
    Wraps a function so that its vjp can be specified and its invocation can be recorded.
    Essentially this behaves just like the decorator
    :return: the wrapped function object
    """
    def func_wrapped(*args, **kwargs):
        """Graph is constructed here
        1. unwrap data and func
        2. ans = func(data)
        3. rewrap them into container
        """

        inside_args, trace_id = find_insider_args(args)
        if inside_args:
            x_ = list(args)
            for i, v in [(argnum, box._value) for argnum, box in inside_args]:
                x_[i] = v
            argvals = tuple(x_)
            parents = tuple(box._node for _, box in inside_args)
            argnums = tuple(argnum for argnum, _ in inside_args)
            ans = func_wrapped(*argvals, **kwargs)
            node = Node(ans, func_wrapped, argvals, kwargs, argnums, parents)
            return Container.new_container(ans, trace_id, node)
        else:
            return f_raw(*args, **kwargs)

    func_wrapped.__doc__ = 'name: ' + f_raw.__name__
    return func_wrapped


def notracefunc_wrapper(f_raw):
    """
    Mainly used for wrap non-diff function.
    :return: the wrapped function object
    """
    def func_wrapped(*args, **kwargs):
        """
        :return: result are not in the container, so it's not in graph neither.
        """
        argvals = map(lambda x: x._value
                      if type(x) in Container.types else x, args)
        return f_raw(*argvals, **kwargs)

    func_wrapped.__doc__ = 'name: ' + f_raw.__name__
    return func_wrapped


def find_insider_args(args):
    top_trace_id = -1
    top_containers = []
    for argnum, arg in enumerate(args):
        if type(arg) in Container.types:
            if arg._trace_id > top_trace_id:
                top_containers = [(argnum, arg)]
                top_trace_id = arg._trace_id
            elif arg._trace_id == top_trace_id:
                top_containers.append((argnum, arg))
    return top_containers, top_trace_id


class TraceStack(object):
    """
    Tracks orders of multi-order diff has been called.
    """
    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1


trace_stack = TraceStack()
