import numpy as np
import matplotlib.pyplot as plt 
from openai_utils import gpt_func
import json

class Neuron:
    def __init__(self, n_inputs, lr) -> None:
        self.weights = str(list(np.random.uniform(-1, 1, (n_inputs,))))
        self.bias = str(list(np.random.uniform(-1, 1, (1,))))
        self.lr = lr

    def sigmoid(self, x):
        return gpt_func("SIGMOID", x)

    def sigmoid_gradient(self, x):
        return gpt_func("SIGMOID GRADIENT", x)
        

    def forward(self, x):
        self.input = x
        self.z = gpt_func("wx + b with inputs ordererd left to right: ", [self.weights, x, self.bias])
        self.output = gpt_func("SIGMOID, ", self.z)
        return self.output

    def backward(self, d_out):
        d_z = gpt_func('MULTIPLY, ', [d_out, gpt_func("SIGMOID GRADIENT, ", self.z)])
        d_input = gpt_func('MULTIPLY, ', [d_z, self.weights])
        self.weights = gpt_func('GRADIENT DESCENT UPDATE: w-lr*d*x', [self.weights, self.lr, d_z, self.input])
        self.bias = gpt_func('GRADIENT DESCENT UPDATE: b-lr*d', [self.bias, self.lr, d_z])
        return d_input


class Network:
    def __init__(self, lr):
        self.hidden = [Neuron(2, lr) for _ in range(4)]
        self.out = Neuron(4, lr)

    def forward(self, x):
        hidden = np.array([json.loads(h.forward(x))[0] for h in self.hidden])
        return self.out.forward(hidden)

    def loss(self, y, y_hat):
        return gpt_func('SQUARED LOSS, ', [y, y_hat])

    def loss_gradient(self, y, y_hat):
        return gpt_func('SQUARED LOSS GRADIENT ', [y, y_hat])

    def backwards(self, y, y_hat):
        d_loss = self.loss_gradient(y, y_hat)
        d_hidden = self.out.backward(d_loss)
        for i, h in enumerate(self.hidden):
            h.backward(d_hidden[i])


XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_LABELS = np.array([0, 1, 1, 0])

if __name__ == "__main__":
    net = Network(lr=10)
    losses = []

    for epoch in range(250):
        total_loss = 0.0
        for x, y in zip(XOR_INPUTS, XOR_LABELS):
            y_hat = net.forward(x)
            total_loss += float(json.loads(str(net.loss(y, y_hat))))
            net.backwards(y, y_hat)
        losses.append(total_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    print("\nResults:")
    for x, y in zip(XOR_INPUTS, XOR_LABELS):
        loads = json.loads(str(net.forward(x)))
        if type(loads) == int:
            y_hat = loads
        else:
            y_hat = loads[0]
        print(f"  {x} -> {int(y_hat >= 0.5)} (expected {y})")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.show()
