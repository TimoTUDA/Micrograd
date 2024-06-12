# Micrograd from Scratch
This repository contains a Jupyter notebook that implements a simple, educational version of the micrograd library from scratch. The implementation focuses on creating a small-scale automatic differentiation engine, primarily designed to be easy to understand and educational for those learning about backpropagation and computational graphs.

## Overview
The notebook walks through the process of building a minimal gradient descent library that includes key components such as value objects, basic arithmetic operations, and the backward pass for gradient computation. This project is inspired by the micrograd library, aiming to provide a deeper understanding of how automatic differentiation works at a low level.

## Gradient Computation
The backward pass is implemented using the backward method of the Value class. This method builds a topological order of the computational graph and performs the backward pass to compute gradients.
```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```
## Example Usage
Below is an example of how to use the Value class to perform simple operations and compute gradients.
```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
d.backward()
print(f"a.grad: {a.grad}, b.grad: {b.grad}, c.grad: {c.grad}")
```
## Visualizing Decision Boundaries
The notebook also demonstrates how to visualize decision boundaries using a simple neural network built with the Value class. Below is an example plot showing the decision boundaries learned by the network.

![image](https://github.com/TimoTUDA/Micrograd/assets/116888691/0ea773a6-62ae-418d-9d60-990576bc4731)
![image](https://github.com/TimoTUDA/Micrograd/assets/116888691/ce10463f-2a42-45f1-be77-59d8e45a4294)


## Conclusion
This project provides a hands-on approach to understanding the basics of automatic differentiation and gradient descent. By building the micrograd library from scratch, you can gain a deeper insight into how machine learning frameworks like PyTorch and TensorFlow compute gradients and perform backpropagation.

## Note:
A very clear and understandable explanation can be found by Andrej Karpathy at https://www.youtube.com/watch?v=VMj-3S1tku0
