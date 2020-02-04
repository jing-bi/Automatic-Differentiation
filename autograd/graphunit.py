class Node(object):
    """
    :param value: output of fun(*args, **kwargs)
    :param fun: wrapped numpy
    :param args: container positional arguments
    :param kwargs: optional keywords arguments
    :param parent_argnums: index of wrapped value contrainer
    :param parents: Node instances corresponding to parent_argnums.
    """
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        self.recipe = (fun, value, args, kwargs, parent_argnums)

    def initialize_root(self):
        self.parents = []
        self.recipe = (lambda x: x, None, (), {}, [])

    @classmethod
    def new_root(cls):
        """ Used to construct a new start node at the begining of the graph
        This can also be done in the naive way == Node(None,lambda x: x, (), {}, [],[])
        :return: New empty node
        """
        root = cls.__new__(cls)
        root.initialize_root()
        return root


class Container(object):
    """Container is the atom unit in the graph.
    wrapped function are always applied to the contrainer
    :param value: single input value/ value of the node
    :param trace_id: used to track multi-order differential
    :param node: Node instances
    """
    type_mappings = {}
    types = set()
    # when this priority > numpy will treat numpy/arraycontainer as whole == treat arraycontainer as a list not a number
    # difference: treating container as number, np will first boardcast this container into a list of container
    __array_priority__ = 1

    def __init__(self, value, trace_id, node):
        self._value = value
        self._node = node
        self._trace_id = trace_id

    def __str__(self):
        return f"{type(self).__name__} with value {self._value}"

    @classmethod
    def register(cls, value_type):
        """Register a new type of data container with value_type.
        :param cls: Inherits from Box. 'value_type' container such as nparray container
        :param value_type: specific data type such as [np.ndarray,float, np.float64, np.float32]
       """
        Container.types.add(cls)
        Container.type_mappings[value_type] = cls
        Container.type_mappings[cls] = cls

    @classmethod
    def new_container(cls, value, trace_id, node):
        """ Used to construct a new container
        If type of data value is not register, then we cannot do anything about it
        :return: A wrapped value
        """
        try:
            return Container.type_mappings[type(value)](value, trace_id, node)
        except KeyError:
            raise TypeError("Can't differentiate w.r.t. type {}".format(
                type(value)))
