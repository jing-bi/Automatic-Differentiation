import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.backpropagate import vector_jacobian

# from autograd.util import subval


# build a proxy function that can get the derivative of a function
def prime(fun):
    '''
    Given fun, return prime of that fun without specific x_value
    :param fun: single-argument function like tanh()
    :return: prime func
    '''
    def funcprime(x_value):
        '''
        :param x_value: we need x to evaluate gradient wrt to this particular argument.
        :return:
            vector-Jacobian product of fun() with all ones
            The argument for vjp is gradient from behind node, here is one
        '''
        vjp, ans = vector_jacobian(fun, x_value)

        return vjp(np.ones_like(ans))

    return funcprime


# Example 1: tanh diff
def tanh(x):
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))


x = np.linspace(-7, 7, 200)
for i in range(6):
    plt.plot(x, eval('prime(' * i + 'tanh' + ')' * i + '(x)'))
plt.axis('off')
plt.show()


# Example 2: compound function
def f(x):
    def g(y):
        return x * y

    return prime(g)(x)


y = prime(f)(5.)
print(y)
