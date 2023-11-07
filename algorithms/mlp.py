import numpy as np
from algorithms.base import BaseAlgorithm
from algorithms.utils import sigmoid, sigmoid_derivative, mean_squared_error

class MLP(BaseAlgorithm):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.layers = self.config['layers']
        self.weights = []
        self.biases = []
        self._initialize_network()
        
    def _initialize_network(self):
        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            weight = np.random.rand(self.layers[i], self.layers[i + 1])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _forward_pass(self, X):
        # Compute the forward pass, storing intermediate results for use in backpropagation
        activations = [X]
        inputs = X
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(inputs, weight) + bias
            outputs = sigmoid(z)
            activations.append(outputs)
            inputs = outputs
        return activations
    
    def _backward_pass(self, y_true, activations):
        # Compute the backward pass using the stored results from the forward pass
        errors = [mean_squared_error(y_true, activations[-1])]
        delta = (y_true - activations[-1]) * sigmoid_derivative(activations[-1])
        deltas = [delta]
        
        # Iterate backwards through layers, starting from the last layer
        for i in reversed(range(len(self.weights))):
            delta = np.dot(deltas[0], self.weights[i].T) * sigmoid_derivative(activations[i])
            deltas.insert(0, delta)
        
        # Calculate gradients for weights and biases
        gradients = []
        for i in range(len(self.weights)):
            gradient = np.dot(activations[i].T, deltas[i + 1])
            gradients.append(gradient)
        
        bias_gradients = [np.sum(delta, axis=0, keepdims=True) for delta in deltas[1:]]
        return gradients, bias_gradients

    def _update_weights(self, gradients, bias_gradients, learning_rate):
        # Update the weights and biases using the gradient descent rule
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * gradients[i]
            self.biases[i] += learning_rate * bias_gradients[i]
    
    def fit(self, X, y, learning_rate=0.1, epochs=1000):
        # Train the network for a fixed number of epochs
        for epoch in range(epochs):
            activations = self._forward_pass(X)
            gradients, bias_gradients = self._backward_pass(y, activations)
            self._update_weights(gradients, bias_gradients, learning_rate)
            
            # Print out the loss every 100 epochs
            if epoch % 100 == 0:
                y_pred = activations[-1]
                loss = mean_squared_error(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        # Make predictions for a batch of instances
        activations = self._forward_pass(X)
        y_pred = activations[-1]
        return y_pred
    
    def __str__(self) -> str:
        return f"Multilayer Perceptron with configuration: {self.config}"
