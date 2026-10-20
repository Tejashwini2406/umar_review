"""Federated Learning simulation for IoT IDS.

Implements FedAvg (Federated Averaging) with dataset-as-client mapping.
Each IoT dataset acts as an independent FL client that trains locally
and shares only model parameters with a central aggregation server.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (AISTATS 2017).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FLClientConfig:
    """Configuration for a single FL client (one IoT domain/dataset)."""
    client_id: str
    local_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001


@dataclass
class FLServerConfig:
    """Configuration for the FL aggregation server."""
    num_rounds: int = 15
    aggregation_strategy: str = "fedavg"  # Only FedAvg implemented
    seed: int = 42


@dataclass
class FLRoundMetrics:
    """Metrics for a single FL communication round."""
    round_num: int
    per_client_loss: dict[str, float] = field(default_factory=dict)
    per_client_train_samples: dict[str, int] = field(default_factory=dict)
    global_eval_metrics: dict[str, Any] = field(default_factory=dict)


class FLClient:
    """Simulated FL client that trains a local model on its own dataset partition.

    In a real FL deployment, each client would be a separate IoT edge device
    or gateway. Here, each client holds one dataset's training data.
    No raw data leaves the client — only model state_dict is shared.
    """

    def __init__(
        self,
        config: FLClientConfig,
        model_template: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.config = config
        self.model = copy.deepcopy(model_template)
        self.X_train = X_train
        self.y_train = y_train
        self._setup_data()

    def _setup_data(self):
        """Creates a DataLoader from the client's local data."""
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        self.loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def receive_global_model(self, global_state_dict: dict):
        """Receives updated global model parameters from server."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def local_train(self) -> tuple[dict, float, int]:
        """Trains the local model for config.local_epochs epochs.

        Returns:
            Tuple of (updated state_dict, average loss, number of training samples)
        """
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            for bx, by in self.loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return self.model.state_dict(), avg_loss, len(self.X_train)


class FLServer:
    """Simulated FL aggregation server implementing FedAvg.

    Coordinates communication rounds: distributes global model to clients,
    collects local updates, and performs weighted averaging of parameters.
    """

    def __init__(
        self,
        config: FLServerConfig,
        global_model: nn.Module,
        clients: list[FLClient],
    ):
        self.config = config
        self.global_model = global_model
        self.clients = clients
        self.round_history: list[FLRoundMetrics] = []

    def _fedavg_aggregate(
        self,
        client_state_dicts: list[dict],
        client_weights: list[float],
    ) -> dict:
        """Performs FedAvg: weighted average of client model parameters.

        Args:
            client_state_dicts: List of state_dicts from each client.
            client_weights: Normalized weights (typically proportional to dataset size).

        Returns:
            Aggregated global state_dict.
        """
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        aggregated = {}
        for key in client_state_dicts[0]:
            aggregated[key] = sum(
                w * sd[key].float() for w, sd in zip(normalized_weights, client_state_dicts)
            )
        return aggregated

    def run_round(self, round_num: int) -> FLRoundMetrics:
        """Executes one FL communication round.

        1. Distribute global model to all clients
        2. Each client trains locally
        3. Collect local updates
        4. Aggregate via FedAvg
        5. Update global model
        """
        metrics = FLRoundMetrics(round_num=round_num)
        global_state = copy.deepcopy(self.global_model.state_dict())

        client_updates = []
        client_weights = []

        for client in self.clients:
            # Step 1: Send global model to client
            client.receive_global_model(global_state)

            # Step 2: Client trains locally
            updated_state, avg_loss, n_samples = client.local_train()

            client_updates.append(updated_state)
            client_weights.append(float(n_samples))

            metrics.per_client_loss[client.config.client_id] = avg_loss
            metrics.per_client_train_samples[client.config.client_id] = n_samples

        # Step 3: Aggregate via FedAvg
        aggregated_state = self._fedavg_aggregate(client_updates, client_weights)
        self.global_model.load_state_dict(aggregated_state)

        self.round_history.append(metrics)
        return metrics

    def run_training(
        self,
        eval_fn: Any | None = None,
        verbose: bool = True,
    ) -> list[FLRoundMetrics]:
        """Runs the complete FL training loop for config.num_rounds rounds.

        Args:
            eval_fn: Optional callable(model) -> dict that evaluates the global model.
            verbose: If True, prints per-round summaries.

        Returns:
            List of per-round metrics.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        for r in range(1, self.config.num_rounds + 1):
            metrics = self.run_round(r)

            if eval_fn is not None:
                self.global_model.eval()
                eval_results = eval_fn(self.global_model)
                metrics.global_eval_metrics = eval_results

            if verbose:
                avg_loss = np.mean(list(metrics.per_client_loss.values()))
                line = f"  Round {r:3d}/{self.config.num_rounds} | Avg Client Loss: {avg_loss:.4f}"
                if metrics.global_eval_metrics:
                    auc = metrics.global_eval_metrics.get("mean_auc", "N/A")
                    line += f" | Global AUC: {auc}"
                print(line)

        return self.round_history

    def save_history(self, path: Path):
        """Saves training history as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(m) for m in self.round_history]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def get_global_model(self) -> nn.Module:
        """Returns the current global model."""
        return self.global_model
