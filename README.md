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