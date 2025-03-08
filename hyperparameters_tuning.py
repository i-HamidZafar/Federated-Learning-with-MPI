from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from mpi4py import MPI
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FederatedMLPLearning:
    def __init__(self, X, y, rank, size):
        self.rank = rank
        self.size = size
        self.X_local, self.y_local = self._split_data(X, y, rank, size)  # Split data locally
        self.local_model = None  # Initialize local model placeholder

    def _split_data(self, X, y, rank, size):
        num_samples = len(X)
        chunk_size = max(1, num_samples // size)  # Ensure at least 1 sample per rank
        start = rank * chunk_size
        end = start + chunk_size if rank != size - 1 else num_samples
        return X[start:end], y[start:end]

    def federated_averaging(self, comm):
        """
        Perform federated averaging of model weights.
        """
        # Collect local model weights
        if self.local_model is not None:
            local_weights = self.local_model.coefs_ + self.local_model.intercepts_
        else:
            local_weights = None
        all_weights = comm.gather(local_weights, root=0)

        # Average weights at root process
        if self.rank == 0:
            global_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*all_weights) if layer_weights[0] is not None]
        else:
            global_weights = None

        # Broadcast global weights to all processes
        global_weights = comm.bcast(global_weights, root=0)

        # Update local model with global weights
        if self.local_model is not None:
            self._set_weights(global_weights)

    def _set_weights(self, global_weights):
        """
        Update the local model's weights.
        """
        split_index = len(self.local_model.coefs_)
        self.local_model.coefs_ = global_weights[:split_index]
        self.local_model.intercepts_ = global_weights[split_index:]

    def _compute_metrics(self, y_true, y_pred):
        """
        Compute classification metrics.
        """

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_and_evaluate(self, comm, rounds=1):
        """
        Federated training and evaluation with hyperparameter optimization.
        """
        # Hyperparameter search space
        hidden_layer_combinations = [(50,), (100,), (50, 50), (100, 50), (50, 100), (50, 200), (50, 400), (100,400),(400,200),(200,400)]
        learning_rates = [0.002, 0.005, 0.004, 0.008 ,0.01, 0.02, 0.05, 0.1, 0.2]

        best_accuracy_global = 0
        best_params_global = None
        best_global_metrics = None

        for round in range(rounds):
            if self.rank == 0:
                print(f"Training Round {round + 1}...\n{'-' * 50}")
            for hl in hidden_layer_combinations:
                for lr in learning_rates:
                    if len(self.X_local) == 0:
                        print(f"[Rank {self.rank}] No data assigned. Skipping this round.", flush=True)
                        continue  # Skip training for ranks with no data

                    # Train local model
                    self.local_model = MLPClassifier(activation='relu', hidden_layer_sizes=hl, learning_rate_init=lr, max_iter=400, random_state=42)
                    self.local_model.fit(self.X_local, self.y_local)

                    # Evaluate local metrics before federated averaging
                    y_pred_local = self.local_model.predict(self.X_local)
                    local_metrics = self._compute_metrics(self.y_local, y_pred_local)
                    print("\n\tLOCAL MEASURED RESULTS\n")
                    # Print local metrics
                    print(f"\t[Rank {self.rank}] Local Metrics (Hidden Layers: {hl}, LR: {lr}): {local_metrics}\n", flush=True)
                    print("-" * 50)

                    # Perform federated averaging
                    self.federated_averaging(comm)

                    # Gather local predictions and true values at root
                    all_y_true = comm.gather(self.y_local, root=0)
                    all_y_pred = comm.gather(y_pred_local, root=0)

                    if self.rank == 0:
                        # Aggregate global results
                        y_true_global = np.concatenate(all_y_true)
                        y_pred_global = np.concatenate(all_y_pred)
                        global_metrics = self._compute_metrics(y_true_global, y_pred_global)

                        # Check for the best global model
                        if global_metrics['accuracy'] > best_accuracy_global:
                            best_accuracy_global = global_metrics['accuracy']
                            best_params_global = {'hidden_layer_sizes': hl, 'learning_rate': lr}
                            best_global_metrics = global_metrics
                            best_global_weights = [np.copy(w) for w in self.local_model.coefs_ + self.local_model.intercepts_]

                        # Print global metrics
                        print("\nGLOBAL MEASURED RESULTS")
                        print(f"\t[Rank {self.rank}] Global Metrics (Hidden Layers: {hl}, LR: {lr}): {global_metrics}\n")
                        print("-" * 50)

        if self.rank == 0:
            print("\n\nBest MEASURED RESULTS")
            print("\nBest Global Hyperparameters:", best_params_global)
            print(f"Best Global Metrics: {best_global_metrics}")
            print("\nBest Global Weights:")
            for idx, layer_weights in enumerate(best_global_weights):
                print(f"Layer {idx + 1}: {layer_weights.shape}\n{layer_weights}")



def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load dataset
    data = pd.read_csv("Pakistani_Diabetes_Dataset.csv")
    label_column = 'income'
    if label_column not in data.columns:
        raise KeyError(f"'{label_column}' not found in dataset columns. Available columns: {data.columns.tolist()}")

    # Encode categorical features
    def encode_categorical_features(data):
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        return data, label_encoders

    data, label_encoders = encode_categorical_features(data)

    # Separate features and labels
    X = data.drop(label_column, axis=1).values
    y = data[label_column].values

    # Scale features
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    # Train-test split (only on root process, then distributed)
    if rank == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = None, None, None, None

    # Broadcast train-test data
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Initialize federated trainer
    federated_trainer = FederatedMLPLearning(X_train, y_train, rank, size)

    # Train and evaluate iteratively
    federated_trainer.train_and_evaluate(comm, rounds=1)


if __name__ == "__main__":
    main()
