import math

class Var:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        z = Var(self.value + other.value)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Var(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z

def sin(x):
    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

a = Var(1)
b = Var(2)
c = Var(3)
d = a + b
e = b * c
f = d * e
f.grad_value=1.0

assert abs(x.grad() - (y.value + math.cos(x.value))) <= 1e-15
