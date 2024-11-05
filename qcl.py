import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


class QCLParams:
    def __init__(self):
        # Quantum Circuit Parameters
        self.n_qubits = 4
        self.circuit_depth = 3
        self.n_shots = 2000
        self.n_params = self.n_qubits * self.circuit_depth * 3 + 4

        # Training Parameters
        self.n_epochs = 300
        self.learning_rate = 0.3
        self.batch_size = 4
        self.gradient_epsilon = np.pi / 2  # Parameter shift rule epsilon
        self.min_improvement = 0.001
        self.early_stopping_patience = 20
        self.target_loss = 0.1
        self.lr_decay_rate = 0.97  # Learning rate decrease 3% per epoch. LR*lr_decay_rate
        self.lr_decay_patience = 12


class QCL:
    def __init__(self, params: QCLParams):
        self.params = params
        self.backend = Aer.get_backend('qasm_simulator')

    def input_layer(self, x):
        """Input layer of the quantum circuit. It encodes the input features into the quantum state.
            Because amplitude encoding is used, we apply RY gates to each qubit."""
        qc = QuantumCircuit(self.params.n_qubits)
        for q in range(self.params.n_qubits):
            qc.ry(x[q] * np.pi, q)
        return qc

    def hidden_layer(self, circuit_params):
        """Hidden layer of the quantum circuit. This is variational quantum circuit that is trained to learn the
            representation of the input data."""
        qc = QuantumCircuit(self.params.n_qubits)
        param_idx = 0
        for d in range(self.params.circuit_depth):
            for q in range(self.params.n_qubits):
                qc.ry(circuit_params[param_idx], q)
                param_idx += 1
                qc.rz(circuit_params[param_idx], q)
                param_idx += 1
                qc.ry(circuit_params[param_idx], q)
                param_idx += 1

            # Get all combinations of qubits for entanglement
            qubit_combs = list(combinations(range(self.params.n_qubits), 2))
            for q1, q2 in qubit_combs:  # Entangle each pair of qubits
                qc.cx(q1, q2)

            for q1, q2 in qubit_combs:  # Cross-entangle each pair of qubits
                qc.cx(q2, q1)

        return qc

    def output_layer(self, circuit_params):
        """Output layer of the quantum circuit. It maps the quantum state to the output probabilities for each class."""
        qc = QuantumCircuit(self.params.n_qubits)
        qc.ry(circuit_params[-4], 0)
        qc.rz(circuit_params[-3], 0)
        qc.ry(circuit_params[-2], 1)
        qc.rz(circuit_params[-1], 1)
        return qc

    def create_quantum_circuit(self, x, circuit_params):
        """Create the full quantum circuit by composing the input, hidden, and output layers."""
        qr = QuantumRegister(self.params.n_qubits)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.compose(self.input_layer(x), inplace=True)
        qc.compose(self.hidden_layer(circuit_params), inplace=True)
        qc.compose(self.output_layer(circuit_params), inplace=True)
        # Because there are only 3 classes, we only need 2 qubits for the output
        qc.measure(0, 0)
        qc.measure(1, 1)

        return qc

    def get_expectation(self, circuit_params, x):
        """Get the expectation value of the quantum circuit for a given input x."""
        qc = self.create_quantum_circuit(x, circuit_params)
        #qc.draw('mpl')
        #plt.show()
        job = self.backend.run(qc, shots=self.params.n_shots)
        result = job.result()
        counts = result.get_counts()

        # Convert counts to probabilities for each class
        probs = np.zeros(4)  # For states 00, 01, 10, 11
        total_shots = self.params.n_shots

        for state, count in counts.items():
            idx = int(state, 2)  # Convert binary state to decimal. E.g., '01' -> 1
            probs[idx] = count / total_shots

        # Get first three probabilities for each class because there are only 3 classes
        epsilon = 1e-10  # To prevent log(0)
        return np.clip(probs[:3], epsilon, 1 - epsilon)

    def compute_gradient_and_loss(self, circuit_params, x_batch, y_batch):
        """Compute the gradient of the loss function with respect to the circuit parameters.
        To compute the gradient, parameter-shift rule is used."""
        gradient = np.zeros(self.params.n_params)
        batch_loss = 0
        for x, y in zip(x_batch, y_batch):
            current_pred = self.get_expectation(circuit_params, x)
            sample_loss = -np.sum(y * np.log(current_pred))
            batch_loss += sample_loss
            for param_idx in range(self.params.n_params):
                circuit_params[param_idx] += self.params.gradient_epsilon  # Forward shift
                pred_plus = self.get_expectation(circuit_params, x)
                circuit_params[param_idx] -= 2 * self.params.gradient_epsilon  # Backward shift
                pred_minus = self.get_expectation(circuit_params, x)
                circuit_params[param_idx] += self.params.gradient_epsilon
                gradients = (pred_plus - pred_minus) / 2
                errors = current_pred - y
                class_gradients = gradients * errors
                gradient[param_idx] += np.sum(class_gradients)
        batch_loss /= len(x_batch)
        return gradient, batch_loss

    def calculate_loss(self, circuit_params, x_batch, y_batch):
        """Calculate the cross-entropy loss for the given batch of data."""
        total_loss = 0
        for x, y in zip(x_batch, y_batch):
            pred = self.get_expectation(circuit_params, x)
            total_loss -= np.sum(y * np.log(pred))
        return total_loss / len(x_batch)

    def train(self, x_train, y_train, x_val, y_val):
        """Train the quantum circuit using the training data and validate on the validation data.
        Mini-batch gradient descent is used."""
        circuit_params = np.random.random(self.params.n_params) * 2 * np.pi
        best_params = circuit_params.copy()
        best_loss = float('inf')
        learning_rate = self.params.learning_rate
        no_improvement = 0

        # Lists to store loss values for plotting
        train_losses = []
        val_losses = []

        print("\nStarting Training...")
        start_time = time.time()

        n_samples = len(x_train)
        n_batches = n_samples // self.params.batch_size
        if n_samples % self.params.batch_size != 0:
            n_batches += 1  # For handling the last partial batch

        for epoch in range(self.params.n_epochs):
            # To avoid getting stuck in local minimum, shuffle the training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.params.batch_size
                end_idx = min(start_idx + self.params.batch_size, n_samples)  # Last batch may be smaller
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_size = end_idx - start_idx  # Actual batch size may be smaller for the last batch
                gradient, batch_loss = self.compute_gradient_and_loss(circuit_params, x_batch, y_batch)
                circuit_params = circuit_params - learning_rate * gradient / batch_size  # Apply gradient descent
                epoch_loss += batch_loss

            epoch_loss /= n_batches
            train_losses.append(epoch_loss)  # Append for plotting
            val_loss = self.calculate_loss(circuit_params, x_val, y_val)
            val_losses.append(val_loss)  # Append for plotting

            if val_loss < best_loss - self.params.min_improvement:
                best_loss = val_loss
                best_params = circuit_params.copy()
                no_improvement = 0
            else:
                no_improvement += 1

                if no_improvement >= self.params.lr_decay_patience:
                    learning_rate = learning_rate * self.params.lr_decay_rate
                    print(f"\nLearning rate decayed to: {learning_rate:.6f}")
            if epoch % 10 == 0:
                total_time = time.time() - start_time
                print(f"\nEpoch {epoch}/{self.params.n_epochs}:")
                print(f"Train Loss: {epoch_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print(f"Learning Rate: {learning_rate:.6f}")
                print(f"Total Time: {total_time:.2f}s")

            if no_improvement >= self.params.early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {self.params.early_stopping_patience} epochs.")
                break

            if val_loss < self.params.target_loss:
                print("\nReached target loss value!")
                break

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Over Epochs")
        plt.show()

        return best_params, best_loss

    def predict(self, x, circuit_params):
        """Predict the class labels for the given input data."""
        preds = []
        for x_i in x:
            probs = self.get_expectation(circuit_params, x_i)
            preds.append(np.argmax(probs))
        return np.array(preds)  # Return the index of the highest probability

    def plot_confusion_matrix(self, cm, class_names, dataset_name="Test"):
        """Plot the confusion matrix for the given dataset."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {dataset_name} Set")
        plt.show()

    def evaluate(self, x, y, circuit_params, dataset_name="Test"):
        """Evaluate the model on the given dataset."""
        predictions = self.predict(x, circuit_params)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y_true)
        loss = self.calculate_loss(circuit_params, x, y)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, predictions)
        class_names = ["Class 0", "Class 1", "Class 2"]  # Update based on actual class names

        # Display results
        print(f"\n{dataset_name} Results:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Loss: {loss:.6f}")

        # Plot the confusion matrix
        self.plot_confusion_matrix(cm, class_names, dataset_name=dataset_name)

        return accuracy, loss, predictions

    def save_parameters(self, parameters, filename="trained_params.npy"):
        """Save trained parameters."""
        np.save(filename, parameters)
        print(f"Parameters saved to {filename}")

    def load_parameters(self, filename="trained_params.npy"):
        """Load trained parameters."""
        parameters = np.load(filename)
        print(f"Parameters loaded from {filename}")
        return parameters


def main():
    # Load and preprocess IRIS dataset
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Normalize features
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)

    # One-hot encode labels
    encoder = LabelBinarizer()
    y_encoded = encoder.fit_transform(y)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x_normalized, y_encoded, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # Initialize and train model
    qcl_params = QCLParams()
    qcl = QCL(qcl_params)

    print("Dataset splits:")
    print(f"Training set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    print(f"Test set size: {len(x_test)}")
    print(f"Number of parameters: {qcl_params.n_params}")

    train_model = True
    if train_model:
        optimal_params, final_loss = qcl.train(x_train, y_train, x_val, y_val)

        qcl.save_parameters(optimal_params, filename="trained_params.npy")

        # Evaluate model
        train_accuracy, train_loss, _ = qcl.evaluate(x_train, y_train, optimal_params, "Training")
        val_accuracy, val_loss, _ = qcl.evaluate(x_val, y_val, optimal_params,  "Validation")
        test_accuracy, test_loss, _ = qcl.evaluate(x_test, y_test, optimal_params, "Test")

        print("\nFinal Model Statistics:")
        print(f"Training - Accuracy: {train_accuracy * 100:.2f}%, Loss: {train_loss:.6f}")
        print(f"Validation - Accuracy: {val_accuracy * 100:.2f}%, Loss: {val_loss:.6f}")
        print(f"Test - Accuracy: {test_accuracy * 100:.2f}%, Loss: {test_loss:.6f}")
    else:
        optimal_params = qcl.load_parameters(filename="trained_params.npy")
        test_accuracy, test_loss, _ = qcl.evaluate(x_test, y_test, optimal_params, "Test")
        print(f"Test - Accuracy: {test_accuracy * 100:.2f}%, Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()