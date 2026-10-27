# Neural Network from Scratch using NumPy

A simple feedforward neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset.

The goal of this project was to understand how a neural network works internally by implementing forward propagation, backpropagation, gradient descent, and model evaluation without using a deep learning framework.

## Project Overview

The model takes a handwritten digit image, processes its pixel values through a hidden layer, and predicts which digit (0–9) the image represents.

- **Dataset:** MNIST handwritten digits
- **Input features:** 784 pixel values per image (28 × 28)
- **Hidden layer:** 16 neurons
- **Output layer:** 10 neurons, one for each digit
- **Hidden activation:** ReLU
- **Output activation:** Softmax
- **Loss function:** Cross-entropy
- **Optimization:** Stochastic Gradient Descent (SGD)
- **Implementation:** Python and NumPy

## Architecture

The network consists of three layers:

1. **Input layer:** Receives 784 normalized pixel values.
2. **Hidden layer:** Contains 16 neurons that learn weighted combinations of input features.
3. **Output layer:** Produces 10 scores, converted into probabilities representing the predicted digit.

```text
Input Image (784 pixels)
          |
          v
    Fully Connected
          |
          v
 Hidden Layer (16 neurons)
          |
         ReLU
          |
          v
    Fully Connected
          |
          v
 Output Layer (10 neurons)
          |
        Softmax
          |
          v
 Predicted Digit (0–9)
```

Every neuron in a layer is connected to every neuron in the next layer. The hidden neurons do not receive predefined features from the dataset; their activations are calculated from the input using learned weights and biases.

## 1. Data Preprocessing

The dataset contains a `label` column representing the correct digit and 784 pixel columns representing the image.

The labels are separated from the input features, and pixel values are normalized from the range 0–255 to 0–1.

```python
y = df['label'].to_numpy()

data = df.drop(columns=['label']).to_numpy(dtype=np.float64) / 255.0
```

The labeled dataset is divided into training and testing sets using an 80:20 split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps preserve the class distribution in both sets.

## 2. Weight and Bias Initialization

The network uses randomly initialized weights and zero-initialized biases.

```python
W1 = np.random.randn(784, 16) * 0.01
W2 = np.random.randn(16, 10) * 0.01

b1 = np.zeros(16)
b2 = np.zeros(10)
```

The dimensions are:

| Parameter | Shape | Purpose |
|---|---|---|
| `W1` | (784, 16) | Connects input features to hidden neurons |
| `b1` | (16,) | Biases of hidden neurons |
| `W2` | (16, 10) | Connects hidden neurons to output neurons |
| `b2` | (10,) | Biases of output neurons |

### Why random initialization?

If all weights in a layer are initialized identically, neurons can produce identical outputs and receive identical gradients. They may continue learning the same representations, a problem known as **symmetry**.

Random initialization breaks this symmetry by giving neurons different starting weights.

The multiplication by `0.01` keeps the initial weights small. Zero-initialized biases are fine here because random weights already break the symmetry.

## 3. Forward Propagation

Forward propagation transforms the input into a prediction.

### Hidden layer

The input is multiplied by the first weight matrix, and the hidden-layer biases are added.

\[
Z_1 = XW_1 + b_1
\]

ReLU introduces nonlinearity:

\[
A_1 = \max(0,Z_1)
\]

Negative activations become zero, while positive activations remain unchanged.

### Output layer

The hidden activations are passed to the output layer:

\[
Z_2 = A_1W_2 + b_2
\]

Softmax converts the output logits into probabilities:

\[
P_i = \frac{e^{Z_{2,i}}}{\sum_j e^{Z_{2,j}}}
\]

Subtracting the maximum logit before exponentiation improves numerical stability without changing the resulting probabilities.

The predicted digit is the class with the highest probability.

## 4. Loss Function

The model uses categorical cross-entropy for a single labeled example:

\[
L = -\log(P_y + \epsilon)
\]

Here, \(P_y\) is the predicted probability assigned to the correct digit, and \(\epsilon\) is a small constant used to avoid taking the logarithm of zero.

A high probability for the correct class results in a lower loss. A low probability results in a higher loss.

## 5. Backpropagation

Backpropagation calculates how the loss changes with respect to the network's parameters using the chain rule.

These gradients indicate how each parameter influences the loss. The parameters are then updated in the direction intended to reduce it.

### Output-layer gradient

For softmax combined with cross-entropy:

\[
dZ_2 = P - Y
\]

Here, \(Y\) is a one-hot target vector. In the implementation, the predicted probability of the correct class is reduced by one.

### Gradients for the second layer

\[
dW_2 = A_1^T dZ_2
\]

\[
db_2 = dZ_2
\]

### Gradients for the hidden layer

The output-layer gradient is propagated backward through the second weight matrix:

\[
dA_1 = W_2dZ_2
\]

The derivative of ReLU is then applied:

\[
dZ_1 = dA_1 \odot \mathbf{1}(Z_1>0)
\]

The input-to-hidden weight gradient is:

\[
dW_1 = X^T dZ_1
\]

The hidden bias gradient is:

\[
db_1 = dZ_1
\]

The code uses `np.outer()` because it processes one training example at a time.

## 6. Parameter Updates

The weights and biases are updated using gradient descent:

\[
W \leftarrow W-\eta dW
\]

\[
b \leftarrow b-\eta db
\]

Here, \(\eta\) is the learning rate.

- A learning rate that is too high can cause unstable updates or prevent convergence.
- A learning rate that is too low can make training unnecessarily slow.
- More epochs provide more opportunities to learn, but excessive training can lead to overfitting.

Because parameters are updated after every individual training example, this implementation uses **stochastic gradient descent (SGD)** rather than full-batch gradient descent.

## 7. Model Evaluation

The model is evaluated on a held-out test set that is not used for parameter updates.

The predicted class is obtained using `argmax` over the output probabilities.

Accuracy measures the proportion of correctly classified test examples:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Test Examples}}
\times 100
\]

Test loss is also calculated using cross-entropy to measure the quality of the predicted probabilities.

## Results

Results from the current implementation:

| Metric | Result |
|---|---:|
| Training accuracy | 97.17% |
| Test accuracy | 93.56% |
| Test loss | 0.572148 |
| Training epochs | 100 |

The training loss decreased substantially during training, reaching approximately `0.075776` at epoch 90.

The difference between training and test accuracy suggests a generalization gap. This may indicate some overfitting, although the gap alone does not establish its cause.

These results are from one run and may vary with random initialization and training conditions.

## Key Learnings

- How a feedforward neural network processes input features.
- How weight matrices and biases determine neuron activations.
- Why nonlinear activation functions are necessary.
- How softmax produces a probability distribution over classes.
- How cross-entropy measures prediction error.
- How backpropagation uses the chain rule to calculate gradients.
- How gradient descent updates weights and biases.
- Why random weight initialization helps prevent symmetry between neurons.
- Why a model's training performance can differ from its performance on unseen data.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (if used for visualization)

## Limitations and Future Improvements

Possible improvements include:

- Shuffle training examples at the start of each epoch.
- Experiment with different hidden-layer sizes and architectures.
- Compare learning rates and initialization strategies.
- Plot training and validation loss across epochs.
- Investigate regularization and methods to reduce overfitting.
- Compare the manually implemented gradients with numerical gradient checks.

## Conclusion

This project helped me understand the internal mechanics of a neural network by implementing its forward pass, loss calculation, backpropagation, and parameter updates directly with NumPy.

Rather than relying on a deep learning framework to handle these operations, I implemented the core computations manually to build a practical understanding of how neural networks learn from data.
