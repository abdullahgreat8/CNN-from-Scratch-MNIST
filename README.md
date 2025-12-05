# CNN-from-Scratch-MNIST

A pure NumPy implementation of a Convolutional Neural Network (CNN) for MNIST digit classification, built entirely from scratch without using deep learning frameworks.

## 📋 Description

This project demonstrates a complete CNN implementation using only NumPy and basic Python libraries. It includes custom implementations of convolutional layers, batch normalization, pooling, and fully connected layers to classify handwritten digits from the MNIST dataset.

## 🌟 Features

- **Custom Layer Implementations:**
  - Convolutional Layer with He initialization
  - Batch Normalization (for both conv and FC layers)
  - ReLU Activation
  - Max Pooling
  - Fully Connected (Dense) Layers
  - Softmax Output Layer

- **Training Features:**
  - Mini-batch gradient descent
  - Forward and backward propagation
  - Cross-entropy loss function
  - Performance visualization (loss and accuracy plots)
  - Prediction visualization

## 🏗️ Network Architecture

```
Input (28×28×1)
    ↓
Conv Layer (3×3, 16 filters) → Batch Norm → ReLU
    ↓
Conv Layer (3×3, 32 filters) → Batch Norm → ReLU
    ↓
Max Pooling (2×2, stride=2)
    ↓
Flatten
    ↓
Fully Connected (128 neurons) → Batch Norm → ReLU
    ↓
Fully Connected (10 neurons) → Softmax
```

## 📦 Requirements

```
numpy
matplotlib
scikit-learn
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CNN-from-Scratch-MNIST.git
cd CNN-from-Scratch-MNIST
```

2. Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

## 💻 Usage

Run the training script:
```bash
python A2.py
```

The script will:
1. Download and load the MNIST dataset
2. Train the CNN for 3 epochs (default)
3. Display training progress with loss and accuracy
4. Generate performance visualizations:
   - `model_graph.png` - Training/test loss and accuracy curves
   - `pred.png` - Sample predictions vs true labels

## 📊 Output

The model generates two visualization files:
- **model_graph.png**: Loss and accuracy curves over epochs
- **pred.png**: 3×3 grid showing sample predictions with true labels

## 🔧 Customization

You can modify training parameters in the `main()` function:

```python
train_losses, test_losses, train_accuracies, test_accuracies = cnn.train(
    X_train_small,
    y_train_small,
    X_test_small,
    y_test_small,
    epochs=3,           # Number of epochs
    batch_size=32,      # Batch size
    learning_rate=0.01  # Learning rate
)
```

## 📈 Performance

Training on a subset of MNIST (5000 training samples, 1000 test samples):
- Training time: ~varies by hardware
- Test accuracy: Achieves competitive accuracy for a from-scratch implementation
- Real-time batch progress updates every 50 batches

## 🧠 Implementation Details

### Key Components:

1. **ConvLayer**: Implements 2D convolution with He initialization
2. **BatchNormalizationLayer**: Normalizes activations with learnable parameters
3. **MaxPoolLayer**: Reduces spatial dimensions while preserving features
4. **FCLayer**: Standard fully connected layer with bias
5. **ReLULayer**: Non-linear activation function
6. **SoftmaxLayer**: Probability distribution for classification

### Training Process:
- Forward pass through all layers
- Cross-entropy loss calculation
- Backward propagation through layers
- Gradient descent weight updates

## 📝 Notes

- The implementation uses He initialization for convolutional layers
- Batch normalization includes momentum-based running statistics
- The code includes detailed comments for educational purposes
- Training time depends on hardware (CPU-based NumPy operations)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset from OpenML
- Inspired by deep learning fundamentals and educational implementations

---

**Repository Name Suggestion:** `CNN-from-Scratch-MNIST` or `numpy-cnn-mnist`

**Repository Description:** "Pure NumPy implementation of a Convolutional Neural Network for MNIST digit classification. Features custom conv layers, batch normalization, pooling, and backpropagation - all built from scratch without deep learning frameworks."
