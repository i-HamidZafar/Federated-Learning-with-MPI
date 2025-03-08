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
        if self.local_model is not None:
            local_weights = self.local_model.coefs_ + self.local_model.intercepts_
        else:
            local_weights = None
        
        all_weights = comm.gather(local_weights, root=0)

        # Average weights at root process
        if self.rank == 0:
            if all(w is not None for w in all_weights):
                averaged_weights = []
                for layer_weights in zip(*all_weights):
                    averaged_weights.append(np.mean(layer_weights, axis=0))
                self.global_weights = averaged_weights
        
        # Broadcast global weights to all processes
        self.global_weights = comm.bcast(self.global_weights, root=0)

        # Update local model with global weights
        if self.local_model is not None:
            self._set_weights(self.global_weights)
        

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
        Federated training and evaluation with proper weight averaging between rounds.
        """
        hidden_layer_combinations = [(50, 400)]
        learning_rates = [0.004]
        self.global_weights = None

        # Initialize the model before rounds
        self.local_model = MLPClassifier(
            activation='relu',
            hidden_layer_sizes=hidden_layer_combinations[0],
            learning_rate_init=learning_rates[0],
            max_iter=300,
            random_state=42
        )
        self.local_model.partial_fit(self.X_local, self.y_local, classes=np.unique(self.y_local))  # Initialize weights

        for round in range(rounds):
            print(f"\n[Rank {self.rank}] Starting Round {round + 1}", flush=True)

            for hl in hidden_layer_combinations:
                for lr in learning_rates:
                    if len(self.X_local) == 0:
                        print(f"[Rank {self.rank}] No data assigned. Skipping this round.", flush=True)
                        continue

                    if round > 0 and self.global_weights is not None:
                        # Apply the global weights to the local model
                        self._set_weights(self.global_weights)
                        print(f"[Rank {self.rank}] Applied global weights at the start of Round {round + 1}", flush=True)

                    # Train the model
                    self.local_model.fit(self.X_local, self.y_local)

                    # Evaluate local metrics
                    y_pred_local = self.local_model.predict(self.X_local)
                    local_metrics = self._compute_metrics(self.y_local, y_pred_local)
                    print(f"[Rank {self.rank}] Local Metrics after training (Round {round + 1}): {local_metrics}", flush=True)

                    # Gather local weights for federated averaging
                    local_weights = self.local_model.coefs_ + self.local_model.intercepts_
                    all_weights = comm.gather(local_weights, root=0)

                    # Average weights at root process
                    if self.rank == 0:
                        averaged_weights = [
                            np.mean([w for w in layer_weights if w is not None], axis=0)
                            for layer_weights in zip(*all_weights)
                        ]
                        self.global_weights = averaged_weights
                        print(f"[Rank {self.rank}] Computed global weights after Round {round + 1}", flush=True)

                    # Broadcast averaged weights to all processes
                    self.global_weights = comm.bcast(self.global_weights, root=0)


                    # Gather metrics for global evaluation
                    all_y_true = comm.gather(self.y_local, root=0)
                    all_y_pred = comm.gather(y_pred_local, root=0)
                    all_local_metrics = comm.gather(local_metrics, root=0)

                    if self.rank == 0:
                        # Compute global metrics
                        y_true_global = np.concatenate(all_y_true)
                        y_pred_global = np.concatenate(all_y_pred)
                        global_metrics = self._compute_metrics(y_true_global, y_pred_global)

                        print(f"\n[Rank {self.rank}] Global Metrics for Round {round + 1}:")
                        print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
                        print(f"  Precision: {global_metrics['precision']:.4f}")
                        print(f"  Recall: {global_metrics['recall']:.4f}")
                        print(f"  F1: {global_metrics['f1']:.4f}")
                        print("-" * 50)

            # Synchronize before next round
            comm.Barrier()

        if self.rank == 0:
            print("\nFinal Global Weight Statistics:")
            for idx, weights in enumerate(self.global_weights):
                print(f"Layer {idx + 1} - Shape: {weights.shape}")
                print(f"Mean: {np.mean(weights):.6f}, Std: {np.std(weights):.6f}")

        if round == 2:
            exit()


def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load dataset
    data = pd.read_csv("balanced_income_data.csv")
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
    federated_trainer.train_and_evaluate(comm, rounds=5)


if __name__ == "__main__":
    main()
