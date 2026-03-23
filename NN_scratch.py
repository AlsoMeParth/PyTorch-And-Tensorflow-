## Implemeting a neural network from scratch in numpy
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = np.random.randn(input_size, hidden_size)  * np.sqrt(1 / input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)

        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x* (1-x)

    def forward(self, x):
        self.hidden = self.sigmoid(np.dot(x, self.weights1) +self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, x, y, learning_rate):
        self.output_error = self.output - y
        self.output_delta = self.output_error * self.sigmoid_derivative(self.output)
        self.hidden_error = self.output_delta.dot(self.weights2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden)
        self.weights2 -= self.hidden.T.dot(self.output_delta) * learning_rate
        self.bias2 -= np.sum(self.output_delta, axis=0) * learning_rate
        self.weights1 -= x.T.dot(self.hidden_delta) * learning_rate
        self.bias1 -= np.sum(self.hidden_delta, axis=0) * learning_rate
    
    def train(self, x, y, epochs, learning_rate):
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            print(f"Epoch {i+1} : Loss {self.loss(y, self.output)}")

    def loss(self, y, output):
        return np.mean(np.square(y - output))
    
    def predict(self, x):
        return self.forward(x)
    def save_weights(self, filename):
        np.savez('weights.npz', weights1=self.weights1, weights2=self.weights2, bias1=self.bias1, bias2=self.bias2)
    def load_weights(self, filename = 'weights.npz'):
        data = np.load(filename)
        self.weights1 = data['weights1']
        self.weights2 = data['weights2']
        self.bias1 = data['bias1']
        self.bias2 = data['bias2']
    def accuracy(self, y, output):
        preds = (output >= 0.5).astype(int)
        return np.mean(y == preds)

nn = NeuralNetwork(2, 4, 1)
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
nn.train(x, y, 1000, 0.5)
print(nn.predict(x))
print(nn.accuracy(y, nn.predict(x)))
print(nn.loss(y, nn.predict(x)))
print(nn.weights1)
print(nn.weights2)
print(nn.bias1)
print(nn.bias2)
print(nn.accuracy(y, nn.predict(x)))