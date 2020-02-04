from __future__ import absolute_import
from . import opoverwritting as anp
from autograd.backpropagate import defvjp

defvjp(anp.negative, lambda g, ans, x: -g)
defvjp(anp.exp, lambda g, ans, x: ans * g)
defvjp(anp.log, lambda g, ans, x: g / x)
defvjp(anp.tanh, lambda g, ans, x: g / anp.cosh(x)**2)
defvjp(anp.sinh, lambda g, ans, x: g * anp.cosh(x))
defvjp(anp.cosh, lambda g, ans, x: g * anp.sinh(x))

defvjp(anp.add, lambda g, ans, x, y: unbroadcast(x, g),
       lambda g, ans, x, y: unbroadcast(y, g))
defvjp(anp.multiply, lambda g, ans, x, y: unbroadcast(x, y * g),
       lambda g, ans, x, y: unbroadcast(y, x * g))
defvjp(anp.subtract, lambda g, ans, x, y: unbroadcast(x, g),
       lambda g, ans, x, y: unbroadcast(y, -g))
defvjp(anp.divide, lambda g, ans, x, y: unbroadcast(x, g / y),
       lambda g, ans, x, y: unbroadcast(y, -g * x / y**2))
defvjp(anp.true_divide, lambda g, ans, x, y: unbroadcast(x, g / y),
       lambda g, ans, x, y: unbroadcast(y, -g * x / y**2))
defvjp(
    anp.power,
    lambda g, ans, x, y: unbroadcast(x, g * y * x**anp.where(y, y - 1, 1.)),
    lambda g, ans, x, y: unbroadcast(y,
                                     g * anp.log(anp.where(x, x, 1.)) * x**y))


def unbroadcast(target, g, broadcast_idx=0):
    while anp.ndim(g) > anp.ndim(target):
        g = anp.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(anp.shape(target)):
        if size == 1:
            g = anp.sum(g, axis=axis, keepdims=True)
    if anp.iscomplexobj(g) and not anp.iscomplex(target):
        g = anp.real(g)
    return g
