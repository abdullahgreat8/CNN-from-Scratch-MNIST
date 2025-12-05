import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml

def train_test_split(X, y, test_size=0.2):
    total_samples = X.shape[0]
    n_test = int(total_samples * test_size)
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    for_train = indices[:n_test]
    for_test = indices[n_test:]
    X_train = X[for_test]
    X_test = X[for_train]
    y_train = y[for_test]
    y_test = y[for_train]
    return X_train, X_test, y_train, y_test
def load_mnist_dataset():
    print("Loading Dataset")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    # print(mnist)
    X = (
        mnist.data.astype("float32") / 255.0
    )  # We are only concerned with the data representing 784 pixels per image and each pixel has values ranging from 0  to 255
    # print(X) #X is a pandas dataframe
    y = mnist.target.astype(
        "int"
    )  # The target contains the actual answer of the digit from 0-9 in the image
    # print(y)#y is a pandas series
    X = X.to_numpy().reshape(-1, 28, 28, 1)
    # print(X)
    y_one_hot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        y_one_hot[i, y[i]] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, 0.2)
    return X_train, X_test, y_train, y_test
def relu(x):
    return np.maximum(0, x)
def relu_der(x):
    return np.where(x > 0, 1, 0)
def softmax(x):
    exp_x = np.exp(x - np.max(x, 1, True))
    return exp_x / np.sum(exp_x, 1, True)
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, 1)])
    return np.sum(log_likelihood) / m
class ConvLayer:
    def __init__(self, input_shape, kernel_size, num_filters):
        self.input_height, self.input_width, self.input_channels = input_shape
        self.bias = np.zeros((1, 1, 1, num_filters))
        self.filters = np.random.randn(kernel_size, kernel_size, self.input_channels, num_filters)*np.sqrt(2.0 /(kernel_size* kernel_size* self.input_channels))
        self.output_width = self.input_width - kernel_size + 1
        self.output_height = self.input_height - kernel_size + 1
        self.output_shape = (self.output_height, self.output_width, num_filters)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
    def forward(self, input_data):
        self.input = input_data
        batch_size = input_data.shape[0]
        self.pre_activation = np.zeros((batch_size, self.output_height, self.output_width, self.num_filters))
        for i in range(self.output_height):
            for j in range(self.output_width):
                patch = input_data[:, i : i + self.kernel_size, j : j + self.kernel_size, :]
                for k in range(self.num_filters):
                    self.pre_activation[:, i, j, k] = (np.sum(patch * self.filters[:, :, :, k], (1, 2, 3))+ self.bias[0, 0, 0, k])
        return self.pre_activation
    def backward(self, d_output, learning_rate):
        batch_size = d_output.shape[0]
        d_input = np.zeros_like(self.input)
        d_filters = np.zeros_like(self.filters)
        d_bias = np.zeros_like(self.bias)
        for i in range(self.output_height):
            for j in range(self.output_width):
                patch = self.input[
                    :, i : i + self.kernel_size, j : j + self.kernel_size, :
                ]
                for k in range(self.num_filters):
                    d_filters[:, :, :, k] += np.sum(
                        patch
                        * d_output[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis],
                        0,
                    )
                    d_bias[0, 0, 0, k] += np.sum(d_output[:, i, j, k])

        for i in range(self.output_height):
            for j in range(self.output_width):
                for k in range(self.num_filters):
                    d_input[:, i : i + self.kernel_size, j : j + self.kernel_size, :] += (self.filters[:, :, :, k]* d_output[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis])
        self.filters -= learning_rate * d_filters / batch_size
        self.bias -= learning_rate * d_bias / batch_size
        return d_input
class BatchNormalizationLayer:
    def __init__(self, input_shape, is_conv=False):
        self.is_conv = is_conv
        self.momentum = 0.9
        self.epsilon = 1e-5
        if is_conv:
            self.channels = input_shape[2]
            self.gamma = np.ones((1, 1, 1, self.channels))
            self.beta = np.zeros((1, 1, 1, self.channels))
            self.running_mean = np.zeros((1, 1, 1, self.channels))
            self.running_var = np.ones((1, 1, 1, self.channels))
        else:
            self.neurons = input_shape
            self.gamma = np.ones((1, self.neurons))
            self.beta = np.zeros((1, self.neurons))
            self.running_mean = np.zeros((1, self.neurons))
            self.running_var = np.ones((1, self.neurons))
    def forward(self, input_data, training=True):
        self.input = input_data
        if self.is_conv:
            # N, H, W, C = input_data.shape
            if training:
                self.batch_mean = np.mean(input_data, (0, 1, 2), True)
                self.batch_var = np.var(input_data, (0, 1, 2), True)
                self.running_mean = (self.momentum * self.running_mean+ (1 - self.momentum) *self.batch_mean)
                self.running_var = (self.momentum * self.running_var+(1 - self.momentum) *self.batch_var)
                self.normalized = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            else:
                self.normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * self.normalized + self.beta
        else:
            if training:
                self.batch_mean = np.mean(input_data, 0, True)
                self.batch_var = np.var(input_data, 0, True)
                self.running_mean = (self.momentum * self.running_mean+ (1 - self.momentum) * self.batch_mean)
                self.running_var= (self.momentum * self.running_var+ (1 - self.momentum) * self.batch_var)
                self.normalized= (input_data - self.batch_mean)/np.sqrt(self.batch_var+ self.epsilon)
            else:
                self.normalized = (input_data - self.running_mean) /np.sqrt(self.running_var + self.epsilon)
            return self.gamma * self.normalized + self.beta

    def backward(self, d_output, learning_rate):
        if self.is_conv:
            N, H, W, C = self.input.shape
            d_gamma = np.sum(d_output * self.normalized, (0, 1, 2), True)
            d_beta = np.sum(d_output, (0, 1, 2), True)
            d_normalized = d_output * self.gamma
            d_var = np.sum(d_normalized* (self.input - self.batch_mean)*-0.5*np.power(self.batch_var + self.epsilon, -1.5),(0, 1, 2),True,)
            d_mean = np.sum(d_normalized * -1 / np.sqrt(self.batch_var + self.epsilon),(0, 1, 2),True,) + d_var * np.mean(-2 * (self.input - self.batch_mean), (0, 1, 2), True)
            d_input = (d_normalized / np.sqrt(self.batch_var + self.epsilon)+ d_var * 2 * (self.input - self.batch_mean) / (N * H * W)+ d_mean / (N * H * W))
        else:
            d_gamma = np.sum(d_output * self.normalized, 0, True)
            d_beta = np.sum(d_output, 0, True)
            d_normalized = d_output * self.gamma
            N = self.input.shape[0]
            d_var = np.sum(d_normalized* (self.input - self.batch_mean)* -0.5*np.power(self.batch_var + self.epsilon, -1.5),0,True,)
            d_mean = np.sum(d_normalized * -1/ np.sqrt(self.batch_var + self.epsilon),0,True,) + d_var * np.mean(-2 * (self.input - self.batch_mean),0,True)
            d_input = (d_normalized / np.sqrt(self.batch_var + self.epsilon)+ d_var * 2 *(self.input - self.batch_mean)/N+ d_mean/N)
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta
        return d_input
class ReLULayer:
    def __init__(self):
        pass
    def forward(self, input_data, training=None):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output
    def backward(self, d_output, learning_rate=None):
        d_input = d_output * np.where(self.input > 0, 1, 0)
        return d_input

class MaxPoolLayer:
    def __init__(self, input_shape, pool_size=2, stride=2):
        self.input_height, self.input_width, self.input_channels = input_shape
        self.output_width = (self.input_width - pool_size) // stride + 1
        self.output_height = (self.input_height - pool_size) // stride + 1
        self.output_shape = (self.output_height, self.output_width, self.input_channels)

        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
    def forward(self, input_data):
        self.input = input_data
        batch_size = input_data.shape[0]
        self.output = np.zeros((batch_size, self.output_height, self.output_width, self.input_channels))
        self.max_indices = np.zeros((batch_size, self.output_height, self.output_width, self.input_channels, 2),int,)
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = input_data[
                    :,
                    h_start : h_start + self.pool_size,
                    w_start : w_start + self.pool_size,
                    :,
                ]
                for b in range(batch_size):
                    for c in range(self.input_channels):
                        curr_patch = patch[b, :, :, c]
                        max_idx = np.unravel_index(
                            np.argmax(curr_patch), curr_patch.shape
                        )
                        self.max_indices[b, i, j, c] = max_idx
                        max_val = np.max(curr_patch)
                        self.output[b, i, j, c] = max_val
        return self.output

    def backward(self, d_output, learning_rate=None):
        batch_size = d_output.shape[0]
        d_input = np.zeros_like(self.input)
        for b in range(batch_size):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    for c in range(self.input_channels):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        max_i, max_j = self.max_indices[b, i, j, c]
                        d_input[b, h_start + max_i, w_start + max_j, c] += d_output[
                            b, i, j, c
                        ]
        return d_input
class FlattenLayer:
    def __init__(self, input_shape):
        self.output_shape = np.prod(input_shape)
        self.input_shape = input_shape
    def forward(self, input_data):
        self.input = input_data
        self.batch_size = input_data.shape[0]
        self.reshaped_input = input_data.reshape(self.batch_size, -1)
        return self.reshaped_input

    def backward(self, d_output, learning_rate=None):
        self.d_output_reshape = d_output.reshape(self.batch_size, *self.input_shape)
        return self.d_output_reshape
class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
    def forward(self, input_data):
        self.input = input_data
        self.input_length = input_data.shape[0]
        self.output = np.dot(input_data, self.weights) +self.bias
        return self.output
    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output, 0, True)
        d_input = np.dot(d_output, self.weights.T)
        self.weights -= learning_rate * d_weights / self.input_length
        self.bias -= learning_rate * d_bias / self.input_length
        return d_input
class SoftmaxLayer:
    def forward(self, input_data):
        self.input = input_data
        self.output = softmax(input_data)
        return self.output

    def backward(self, d_output, learning_rate=None):  # no need so return as is
        return d_output
class CNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.conv1 = ConvLayer(input_shape, 3, 16)
        self.bn1 = BatchNormalizationLayer(self.conv1.output_shape, is_conv=True)
        self.relu1 = ReLULayer()
        self.conv2 = ConvLayer(self.conv1.output_shape, 3, 32)
        self.bn2 = BatchNormalizationLayer(self.conv2.output_shape, is_conv=True)
        self.relu2 = ReLULayer()
        self.pool = MaxPoolLayer(self.conv2.output_shape)
        self.flatten = FlattenLayer(self.pool.output_shape)
        self.fc = FCLayer(self.flatten.output_shape, 128)
        self.bn3 = BatchNormalizationLayer(128, is_conv=False)
        self.relu3 = ReLULayer()
        self.fc_out = FCLayer(128, 10)
        self.softmax = SoftmaxLayer()
        self.layers = [
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2, self.relu2,
            self.pool, self.flatten,
            self.fc, self.bn3, self.relu3,
            self.fc_out, self.softmax
        ]
        
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            if hasattr(layer, 'forward') and callable(getattr(layer, 'forward')):
                if isinstance(layer, BatchNormalizationLayer):
                    output = layer.forward(output, training)
                else:
                    output = layer.forward(output)
        return output
    
    def backward(self, y_true, learning_rate):
        # batch_size = y_true.shape[0]
        d_output = self.softmax.output - y_true
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward') and callable(getattr(layer, 'backward')):
                d_output = layer.backward(d_output, learning_rate)
    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32, learning_rate=0.01):
        train_accuracies = []
        train_losses = []
        test_accuracies = []
        test_losses = []
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size
        train_start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            for batch in range(num_batches):
                batch_start_time = time.time()
                start_in = batch * batch_size
                end_in = min((batch + 1) * batch_size, num_samples)
                X_batch = X_train_shuffled[start_in:end_in]
                y_batch = y_train_shuffled[start_in:end_in]
                y_pred = self.forward(X_batch, training=True)
                batch_loss = cross_entropy_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                self.backward(y_batch, learning_rate)
                batch_end_time = time.time()
                if batch % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch:{batch}, Loss: {batch_loss:.4f}, Time taken:{batch_end_time - batch_start_time:.4f}")
            epoch_loss /= num_batches
            train_losses.append(epoch_loss)
            test_pred = self.forward(X_test, training=False)
            test_loss = cross_entropy_loss(y_test, test_pred)
            test_losses.append(test_loss)
            train_pred = self.forward(X_train[:1000], training=False)
            train_acc = np.mean(np.argmax(train_pred, 1) == np.argmax(y_train[:1000], 1))
            train_accuracies.append(train_acc)
            test_acc = np.mean(np.argmax(test_pred, 1) == np.argmax(y_test, 1))
            test_accuracies.append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Total time taken: {time.time() - train_start_time:.4f}")
        return train_losses, test_losses, train_accuracies, test_accuracies
    
    def predict(self, X):
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, 1)

def generate_performance(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    predictions,
    X_test_small,
    y_test_small,
):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs. Epoch")
    plt.tight_layout()
    plt.savefig("model_graph.png")
    true_labels = np.argmax(y_test_small, 1)
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(predictions):
            ax.imshow(X_test_small[i].reshape(28, 28), "gray")
            ax.set_title(f"Predicted: {predictions[i]}, True: {true_labels[i]}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig("pred.png")
def main():
    input_shape = (28, 28, 1)
    cnn = CNN(input_shape)
    X_train, X_test, y_train, y_test = load_mnist_dataset()
    print(f"Training: {X_train.shape}")
    print(f"Answers: {y_train.shape}")
    print(f"Testing: {X_test.shape}")
    print(f"Answers: {y_test.shape}")

    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:1000]
    y_test_small = y_test[:1000]
    train_losses, test_losses, train_accuracies, test_accuracies = cnn.train(
        X_train_small,
        y_train_small,
        X_test_small,
        y_test_small,
        3,
        32,
        learning_rate=0.01,
    )
    predictions = cnn.predict(X_test_small)
    generate_performance(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        predictions,
        X_test_small,
        y_test_small,
    )
if __name__ == "__main__":
    main()