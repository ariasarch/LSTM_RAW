import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Initialize gradient matrices
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dWy = np.zeros_like(self.Wy)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)

        # Initialize hidden state and cell state matrices
        self.h_next = np.zeros((hidden_size, self.num_layers))
        self.c_next = np.zeros((hidden_size, self.num_layers))

    def reset_gradients(self):
        self.dWf.fill(0)
        self.dWi.fill(0)
        self.dWc.fill(0)
        self.dWo.fill(0)
        self.dWy.fill(0)
        self.dbf.fill(0)
        self.dbi.fill(0)
        self.dbc.fill(0)
        self.dbo.fill(0)
        self.dby.fill(0)

    def forward(self, x, h_prev, c_prev):
        # print(f'Shape of x: {x.shape}, expected: ({num_samples}, {self.input_size})')
        # print(f'Shape of h_prev: {h_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_prev: {c_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')

        dropout_masks = []
        y_preds = []
        i_list = []
        c_bar_list = []
        f_list = []
        o_list = []

        for t in range(x.shape[0]):  # Iterate over each sample
            xt = x[t].reshape(self.input_size, 1)  # Shape (input_size, 1)
            h_next_t = np.zeros((self.hidden_size, self.num_layers))
            c_next_t = np.zeros((self.hidden_size, self.num_layers))
            h_next_t, mask = dropout(h_next_t, dropout_rate)
            dropout_masks.append(mask)
            i_t = np.zeros((self.hidden_size, self.num_layers))
            c_bar_t = np.zeros((self.hidden_size, self.num_layers))
            f_t = np.zeros((self.hidden_size, self.num_layers))
            o_t = np.zeros((self.hidden_size, self.num_layers))

            for l in range(self.num_layers):
                # print(f'Layer {l+1}:')
                h_prev_l = h_prev[:, l].reshape(self.hidden_size, 1)
                c_prev_l = c_prev[:, l].reshape(self.hidden_size, 1)
                # print(f'  Shape of h_prev_l: {h_prev_l.shape}, expected: ({self.hidden_size}, 1)')
                # print(f'  Shape of c_prev_l: {c_prev_l.shape}, expected: ({self.hidden_size}, 1)')

                concat = np.vstack((h_prev_l, xt))  # Shape (hidden_size + input_size, 1)
                # print(f'  Shape of concat: {concat.shape}, expected: ({self.hidden_size + self.input_size}, 1)')

                f_t[:, l] = sigmoid(np.dot(self.Wf, concat) + self.bf)[:, 0]
                # print(f'  Shape of f_t[:, {l}]: {f_t[:, l].shape}, expected: ({self.hidden_size},)')

                i_t[:, l] = sigmoid(np.dot(self.Wi, concat) + self.bi)[:, 0]
                # print(f'  Shape of i_t[:, {l}]: {i_t[:, l].shape}, expected: ({self.hidden_size},)')

                c_bar_t[:, l] = np.tanh(np.dot(self.Wc, concat) + self.bc)[:, 0]
                # print(f'  Shape of c_bar_t[:, {l}]: {c_bar_t[:, l].shape}, expected: ({self.hidden_size},)')

                c_next_t[:, l] = f_t[:, l] * c_prev_l[:, 0] + i_t[:, l] * c_bar_t[:, l]
                # print(f'  Shape of c_next_t[:, {l}]: {c_next_t[:, l].shape}, expected: ({self.hidden_size},)')

                o_t[:, l] = sigmoid(np.dot(self.Wo, concat) + self.bo)[:, 0]
                # print(f'  Shape of o_t[:, {l}]: {o_t[:, l].shape}, expected: ({self.hidden_size},)')

                h_next_t[:, l] = o_t[:, l] * np.tanh(c_next_t[:, l])
                # print(f'  Shape of h_next_t[:, {l}]: {h_next_t[:, l].shape}, expected: ({self.hidden_size},)')

            yt = np.dot(self.Wy, h_next_t[:, -1].reshape(self.hidden_size, 1)) + self.by
            y_preds.append(yt)
            i_list.append(i_t)
            c_bar_list.append(c_bar_t)
            f_list.append(f_t)
            o_list.append(o_t)

            h_prev = h_next_t
            c_prev = c_next_t

        y_preds = np.array(y_preds).reshape(-1, self.output_size)

        # print(f'Shape of h_next: {h_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_next: {c_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of y_preds: {y_preds.shape}, expected: ({num_samples}, {self.output_size})')

        return y_preds, h_next_t, c_next_t, i_list, c_bar_list, f_list, o_list

    def backward(self, x, y, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list):
        # print(f'Shape of x: {x.shape}, expected: ({num_samples}, {self.input_size})')
        # print(f'Shape of y: {y.shape}, expected: ({num_samples},)')
        # print(f'Shape of y_preds: {y_preds.shape}, expected: ({num_samples}, {self.output_size})')
        # print(f'Shape of h_prev: {h_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_prev: {c_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of dh_next: {dh_next.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of dc_next: {dc_next.shape}, expected: ({self.hidden_size}, {self.num_layers})')

        for t in reversed(range(x.shape[0])):  # Iterate over each sample in reverse order
            dE_dy = y_preds[t] - y[t] # the gradient of the binary cross-entropy loss 
            self.dWy += np.dot(dE_dy.reshape(self.output_size, 1), h_prev[:, -1].reshape(1, self.hidden_size))
            self.dby += dE_dy.reshape(self.output_size, 1)

            dh_next_t = np.zeros((self.hidden_size, self.num_layers))
            dc_next_t = np.zeros((self.hidden_size, self.num_layers))

            for l in reversed(range(self.num_layers)):
                dh = np.dot(self.Wy.T, dE_dy.reshape(self.output_size, 1)) + dh_next_t[:, l].reshape(self.hidden_size, 1)
                dc = dc_next_t[:, l].reshape(self.hidden_size, 1) + dh * o_list[t][:, l].reshape(self.hidden_size, 1) * (1 - np.square(np.tanh(c_prev[:, l].reshape(self.hidden_size, 1))))

                do = dh * np.tanh(c_prev[:, l].reshape(self.hidden_size, 1))
                dc_bar = dh * i_list[t][:, l].reshape(self.hidden_size, 1)
                di = dh * c_bar_list[t][:, l].reshape(self.hidden_size, 1)
                df = dh * c_prev[:, l].reshape(self.hidden_size, 1)

                xt = x[t].reshape(self.input_size, 1)
                concat = np.vstack((h_prev[:, l].reshape(self.hidden_size, 1), xt))

                self.dWf += np.dot(df * sigmoid_derivative(f_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)
                self.dWi += np.dot(di * sigmoid_derivative(i_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)
                self.dWc += np.dot(dc_bar * (1 - np.square(c_bar_list[t][:, l].reshape(self.hidden_size, 1))), concat.T)
                self.dWo += np.dot(do * sigmoid_derivative(o_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)

                self.dbf += df * sigmoid_derivative(f_list[t][:, l].reshape(self.hidden_size, 1))
                self.dbi += di * sigmoid_derivative(i_list[t][:, l].reshape(self.hidden_size, 1))
                self.dbc += dc_bar * (1 - np.square(c_bar_list[t][:, l].reshape(self.hidden_size, 1)))
                self.dbo += do * sigmoid_derivative(o_list[t][:, l].reshape(self.hidden_size, 1))

                dh_next_t[:, l] = dh[:, 0]
                dc_next_t[:, l] = dc[:, 0]

        return self.dWf, self.dWi, self.dWc, self.dWo, self.dWy, self.dbf, self.dbi, self.dbc, self.dbo, self.dby

    def update_weights(self, learning_rate, weight_decay):
        # Updates weights using gradients with L2 regularization.
        self.Wf -= learning_rate * (self.dWf + weight_decay * self.Wf)
        self.Wi -= learning_rate * (self.dWi + weight_decay * self.Wi)
        self.Wc -= learning_rate * (self.dWc + weight_decay * self.Wc)
        self.Wo -= learning_rate * (self.dWo + weight_decay * self.Wo)
        self.Wy -= learning_rate * (self.dWy + weight_decay * self.Wy)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def dropout(x, dropout_rate):

    # Applies dropout by randomly setting a fraction of x to zero.
    if dropout_rate > 0:
        retain_prob = 1 - dropout_rate
        mask = np.random.binomial(1, retain_prob, size=x.shape)
        return x * mask, mask
    return x, np.ones_like(x)

def binary_cross_entropy(y_true, y_pred):
    # Avoid division by zero
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# For example data creation
input_size = 10
hidden_size = 64
output_size = 1
num_samples = 100

# Constants for the linear equation y = mx + b
m = np.random.randn(input_size)  # Random slopes for each feature in x
b = np.random.randn()  # Random intercept

# Generate input data
x = np.random.randn(num_samples, input_size)  # Random input features

# Generate target outputs with a linear relationship passed through a sigmoid function
linear_combination = np.dot(x, m) + b
y_prob = sigmoid(linear_combination)  # Apply sigmoid to convert to probabilities
y = (y_prob > 0.5).astype(int)  # Threshold probabilities to get binary labels

# # Normalize input features
# x_mean = np.mean(x, axis=0)
# x_std = np.std(x, axis=0)
# x_normalized = (x - x_mean) / x_std
# print(f'Shape of x_normalized: {x_normalized.shape}, expected: ({num_samples}, {input_size})')

print(f'Shape of x: {x.shape}, expected: ({num_samples}, {input_size})')
print(f'Shape of y: {y.shape}, expected: ({num_samples},)')

# Initialize LSTM
lstm = LSTM(input_size, hidden_size, output_size)

# Initialize key variables 
num_layers = 64 # Define number of layers 
learning_rate = 0.01 # Define the learning rate
max_grad_norm = 1.0 # Set a threshold for gradient clipping
dropout_rate = 0.2 # Set dropout rate
weight_decay = 0.001 # Set weight decay

# Define the number of epochs and batch size
num_epochs = 10
batch_size = 32

# Initialize a list to store predictions from the final epoch
final_epoch_preds = []
accuracy_over_epochs = []
losses = []  # List to store loss values

# Perform training loop
for epoch in range(num_epochs):
    # Temporary list for the current epoch predictions
    current_epoch_preds = []
    epoch_losses = []
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Shuffle the training data
    indices = np.random.permutation(num_samples)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Iterate over mini-batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        x_batch = x_shuffled[batch_start:batch_end]
        y_batch = y_shuffled[batch_start:batch_end]
        batch_size_actual = y_batch.shape[0]  

        # Initialize hidden state and cell state
        h_prev = np.zeros((hidden_size, lstm.num_layers))
        c_prev = np.zeros((hidden_size, lstm.num_layers))

        # Forward pass
        y_preds, h_next, c_next, i_list, c_bar_list, f_list, o_list = lstm.forward(x_batch, h_prev, c_prev)

        # Store predictions and loss from the current batch
        current_epoch_preds.append(y_preds)
        loss = binary_cross_entropy(y_batch, y_preds.flatten())
        total_loss += loss * batch_size_actual  # Weighting the loss by the batch size

        # Calculate and accumulate accuracy
        batch_predictions = (y_preds.flatten() > 0.5).astype(int)
        total_correct += np.sum(batch_predictions == y_batch)
        total_samples += batch_size_actual

        # Backward pass
        dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby = lstm.backward(x_batch, y_batch, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list)

        # Clipping gradients
        grad_norm = np.sqrt(sum(np.sum(grad**2) for grad in [dWf, dWi, dWc, dWo, dWy]))
        if grad_norm > max_grad_norm:
            clip_coef = max_grad_norm / (grad_norm + 1e-6)  # Avoid division by zero
            dWf, dWi, dWc, dWo, dWy = [clip_coef * grad for grad in [dWf, dWi, dWc, dWo, dWy]]

        # Update weights and biases
        lstm.update_weights(learning_rate=learning_rate, weight_decay=weight_decay)
        lstm.bf -= learning_rate * dbf
        lstm.bi -= learning_rate * dbi
        lstm.bc -= learning_rate * dbc
        lstm.bo -= learning_rate * dbo
        lstm.by -= learning_rate * dby

        # Reset gradients for the next batch
        lstm.reset_gradients()

        # After processing all batches in the current epoch
        if epoch == num_epochs - 1:  # Check if it's the final epoch
            final_epoch_preds = current_epoch_preds  # Only store the final epoch's predictions
    
    # Compute average loss and accuracy for the epoch
    average_epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    losses.append(average_epoch_loss)
    accuracy_over_epochs.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

# Concatenate all predictions from the final epoch
final_y_preds = np.concatenate([pred for pred in final_epoch_preds], axis=0)
print(f"Final predictions shape: {final_y_preds.shape}")

# For final evaluation
def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, np.round(y_pred))
    print("Mean Squared Error:", mse)
    print("Accuracy:", accuracy)

evaluate_predictions(y, final_y_preds.flatten())

