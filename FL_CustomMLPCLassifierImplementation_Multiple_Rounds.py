import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from mpi4py import MPI


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FederatedMLPLearning:

    def __init__(self, X, y, rank, size):
        self.rank = rank
        self.size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split data locally
        self.X_local, self.y_local = self._split_data(X, y, rank, size)

        # Initialize local model
        self.input_size = X.shape[1]
        self.hidden_sizes = [50, 200]
        self.output_size = len(np.unique(y))
        self.local_model = MLPModel(self.input_size, self.hidden_sizes, self.output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.004)
        self.global_weights = None
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

    def _split_data(self, X, y, rank, size, shuffle=True):
        num_samples = len(X)

        # Shuffle the data if shuffle=True
        if shuffle:
            indices = np.random.permutation(num_samples)
            X = X[indices]
            y = y[indices]

        # Split the data based on rank and size
        chunk_size = max(1, num_samples // size)
        start = rank * chunk_size
        end = start + chunk_size if rank != size - 1 else num_samples
        return X[start:end], y[start:end]

    def train_one_epoch(self):
        self.local_model.train()
        X_tensor = torch.tensor(self.X_local, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y_local, dtype=torch.long).to(self.device)

        self.optimizer.zero_grad()
        outputs = self.local_model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def evaluate_local(self):
        self.local_model.eval()
        X_tensor = torch.tensor(self.X_local, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y_local, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.local_model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()

        metrics = {
            'accuracy': accuracy_score(self.y_local, predictions),
            'precision': precision_score(self.y_local, predictions, average='weighted', zero_division=0),
            'recall': recall_score(self.y_local, predictions, average='weighted', zero_division=0),
            'f1': f1_score(self.y_local, predictions, average='weighted', zero_division=0),
        }
        return metrics

    def get_weights(self):
        return {name: param.clone().detach().cpu().numpy() for name, param in self.local_model.named_parameters()}

    def set_weights(self, global_weights):
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                param.copy_(torch.tensor(global_weights[name], dtype=torch.float32).to(self.device))

    def federated_averaging(self, comm):
        # Gather local weights and data sizes
        local_weights = self.get_weights()
        local_data_size = len(self.X_local)
        gathered_weights = comm.gather(local_weights, root=0)
        gathered_sizes = comm.gather(local_data_size, root=0)

        if self.rank == 0:
            # Weighted averaging of model weights based on data size
            total_size = sum(gathered_sizes)
            averaged_weights = {}
            for key in local_weights.keys():
                averaged_weights[key] = sum(
                    (gathered_weights[i][key] * gathered_sizes[i] / total_size) for i in range(len(gathered_weights))
                )
            self.global_weights = averaged_weights

        # Broadcast global weights to all ranks
        self.global_weights = comm.bcast(self.global_weights, root=0)
        self.set_weights(self.global_weights)

    def train_and_evaluate(self, comm, rounds=5, termination_patience=10, tolerance=1e-4):
        
        global_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        termination_count = termination_patience
        prev_metric = None
        stop_signal = False  # Signal to indicate early stopping

        try:
            for round in range(rounds):
                # Broadcast the stop signal to all ranks
                stop_signal = comm.bcast(stop_signal, root=0)
                if stop_signal:
                    if self.rank == 0:
                        print(f"Training stopped early at round {round}.")
                    break

                if self.rank == 0:
                    print(f"\nRound {round + 1}:\n", flush=True)

                # Synchronize before starting a new round
                comm.Barrier()

                # Train the local model
                self.train_one_epoch()

                # Evaluate local metrics
                local_metrics = self.evaluate_local()

                # Synchronize and print local metrics in rank order
                for r in range(self.size):
                    comm.Barrier()  # Synchronize before each rank prints
                    if self.rank == r:
                        print(
                            f"  RANK {self.rank} - Local Metrics (Round {round + 1}): "
                            f"[accuracy: {local_metrics['accuracy']:.4f}, "
                            f"precision: {local_metrics['precision']:.4f}, "
                            f"recall: {local_metrics['recall']:.4f}, "
                            f"f1: {local_metrics['f1']:.4f}]",
                            flush=True
                        )
                    comm.Barrier()  # Synchronize after each rank prints

                # Gather local metrics to rank 0
                gathered_metrics = comm.gather(local_metrics, root=0)

                # Rank 0 calculates and prints global metrics
                if self.rank == 0:
                    avg_metrics = {k: np.mean([m[k] for m in gathered_metrics]) for k in local_metrics.keys()}
                    for k in avg_metrics:
                        global_metrics[k].append(avg_metrics[k])
                    print(
                        f"  Global Metrics (Round {round + 1}): "
                        f"[accuracy: {avg_metrics['accuracy']:.4f}, "
                        f"precision: {avg_metrics['precision']:.4f}, "
                        f"recall: {avg_metrics['recall']:.4f}, "
                        f"f1: {avg_metrics['f1']:.4f}]",
                        flush=True
                    )

                    # Check for early stopping
                    if prev_metric is not None and np.allclose(
                            [avg_metrics[k] for k in avg_metrics],
                            [prev_metric[k] for k in prev_metric],
                            atol=tolerance):
                        termination_count -= 1
                        if termination_count == 0:
                            print(f"Early stopping triggered: No significant change in metrics for {termination_patience} rounds.")
                            stop_signal = True
                    else:
                        prev_metric = avg_metrics
                        termination_count = termination_patience

                # Broadcast the stop signal to all ranks
                stop_signal = comm.bcast(stop_signal, root=0)

                # Perform federated averaging
                self.federated_averaging(comm)

                # Synchronize after federated averaging
                comm.Barrier()

        except Exception as e:
            print(f"Rank {self.rank} encountered an error: {e}", flush=True)
            comm.Abort()  # Abort MPI to ensure proper termination of all ranks

        return global_metrics



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = pd.read_csv("Pakistani_Diabetes_Dataset.csv")
    label_column = 'Outcome'

    if label_column not in data.columns:
        raise KeyError(f"'{label_column}' not found in dataset columns. Available columns: {data.columns.tolist()}")

    def encode_categorical_features(data):
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        return data, label_encoders

    data, label_encoders = encode_categorical_features(data)

    X = data.drop(label_column, axis=1).values
    y = data[label_column].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if rank == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = None, None, None, None

    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    federated_trainer = FederatedMLPLearning(X_train, y_train, rank, size)
    global_metrics = federated_trainer.train_and_evaluate(comm, rounds=300)

if __name__ == "__main__":
    main()
