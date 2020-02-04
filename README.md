# Automatic-Differentiation

Repo for step by step work through the implementation of Automatic Differentiation

## The key component for automatic differentiation can be generalized in to two parts:

1. Tracer
2. How we walk through the graph(forward/backward)

## Approach 1: Naive operation overloading

We can easily design variable in our graph as follow:

```python
class Var:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None
```

The `children` of variable is a Graph representation.
Each time you add an operation to the graph,you will have a new varable with extened graph.
In this way, we have to overloading every operation in our 

## Approach 2: Auto record the graph and backward pass with VJP

### Graph-Unit

1. **Node**: representation for node in graph, its parents are linked list which is the computational graph
2. **Container**: value-tpye related container which used as the atom unit in the forward/backward pass, the advantage of container is that it allow you to convinently costomize your own data container and related function.

### Inference

1. forward pass will traval through the graph and record each invocation of the function
2. Func_wrapper mainly used to contruct a function object which will
   1. unbox container
   2. calculate value with raw function
   3. box a new container
3. the reason for doing this is to build a graph secertly

### Backpropagate

1. vjp is used to get the gradient of the node wrt its function
2. vjp is defined separately by the library you want to use

