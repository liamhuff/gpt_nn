import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_inputs, lr) -> None:
        self.weights = np.random.uniform(-1, 1, (n_inputs,))
        self.bias = np.random.uniform(-1, 1, (1,))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, d_out):
        d_z = d_out * self.sigmoid_gradient(self.z)
        d_input = d_z * self.weights
        self.weights -= self.lr * d_z * self.input
        self.bias -= self.lr * d_z
        return d_input


class Network:
    def __init__(self, lr):
        self.hidden = [Neuron(2, lr) for _ in range(4)]
        self.out = Neuron(4, lr)

    def forward(self, x):
        hidden = np.array([h.forward(x)[0] for h in self.hidden])
        return self.out.forward(hidden)

    def loss(self, y, y_hat):
        return (y - y_hat) ** 2

    def loss_gradient(self, y, y_hat):
        return 2 * (y_hat - y)

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
            total_loss += float(net.loss(y, y_hat))
            net.backwards(y, y_hat)
        losses.append(total_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    print("\nResults:")
    for x, y in zip(XOR_INPUTS, XOR_LABELS):
        y_hat = net.forward(x)
        print(f"  {x} -> {int(y_hat[0] >= 0.5)} (expected {y})")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.show()
