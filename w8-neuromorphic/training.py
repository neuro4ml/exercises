import torch
import pandas as pd
from dataclasses import dataclass
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, random_split

from models import SNNModel
from snntorch.functional.loss import ce_rate_loss
from chip import NeuromorphicChip, calculate_pareto_score


def get_dataloaders(
    batch_size: int = 32,
    train_split: float = 0.8,
):
    data = torch.load("dataset", weights_only=True)
    labels = torch.load("dataset_labels", weights_only=True)
    spike_times = data[..., 0].int().long()
    spikes = torch.nn.functional.one_hot(spike_times, num_classes=100).transpose(1, 2)

    dataset = TensorDataset(spikes, labels)

    # Split into train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
    )

    def collate_fn(batch):
        # Unpack the batch into inputs and targets
        inputs, targets = zip(*batch)
        # Stack and transpose inputs from (batch, time, features) to (time, batch, features)
        inputs = torch.stack(inputs).transpose(0, 1)
        # Stack targets normally
        targets = torch.stack(targets)
        return inputs, targets

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader, dataset


@dataclass
class TrainingMetrics:
    accuracy: float
    energy_usage: float
    epoch: int
    loss: float
    firing_rate: float


class SNNTrainer:
    def __init__(
        self,
        snn: SNNModel,
        learning_rate: float = 0.001,
        lr_gamma: float = 0.9,
        config: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.model = snn.to(device)
        self.chip = NeuromorphicChip()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_gamma
        )

        # Initialize loss function
        self.loss_fn = ce_rate_loss()
        self.metrics_history: list[TrainingMetrics] = []
        self.chip_results: list[pd.DataFrame] = []

    def calculate_accuracy(
        self, spikes: torch.Tensor, target: torch.Tensor
    ) -> tuple[float, float]:
        """
        Calculate accuracy and loss from a rate-based loss
        TODO: Complete this method to return accuracy and loss
        Optional: Implement a temporal time-to-first-spike based loss using snnTorch.
        """

        loss = self.loss_fn(spikes, target).mean()
        acc = None

        raise NotImplementedError("Accuracy not implemented")

        return acc, loss.item()

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        n_epochs: int,
        pbar: tqdm = None,
    ) -> TrainingMetrics:
        self.model.train()
        total_correct = 0
        total_samples = 0
        epoch_energy = 0.0
        epoch_loss = 0.0
        epoch_firing_rate = 0.0

        if pbar is None:
            pbar = tqdm(train_loader, desc="Training: ", leave=False)
            pbar_to_set = pbar
        else:
            pbar_to_set = pbar
            pbar = train_loader

        for batch_idx, (data, target) in enumerate(pbar):
            self.optimizer.zero_grad()

            data = data.float().to(self.device)
            target = target.to(self.device)

            # Forward pass
            spikes, mem = self.model(data)

            # Calculate loss and backward
            loss = self.loss_fn(spikes, target)

            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            acc, loss_val = self.calculate_accuracy(spikes, target)
            firing_rate = spikes.mean().item()

            desc = str(
                f"Epoch {epoch}/{n_epochs} - Batch {batch_idx}/{len(train_loader)}: loss: {loss_val:.3f}, "
                + f"Firing Rate: {firing_rate:.3f}, Acc: {acc:.3f}"
            )
            pbar_to_set.set_postfix_str(desc)

            total_correct += acc * target.size(0)
            total_samples += target.size(0)
            epoch_loss += loss_val
            epoch_firing_rate += firing_rate

        # Calculate epoch metrics
        metrics = TrainingMetrics(
            accuracy=total_correct / total_samples,
            energy_usage=epoch_energy / len(train_loader),
            epoch=epoch,
            loss=epoch_loss / len(train_loader),
            firing_rate=epoch_firing_rate / len(train_loader),
        )
        self.metrics_history.append(metrics)

        # Update learning rate
        self.scheduler.step(
            metrics.loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            else None
        )

        return metrics

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        pbar: tqdm = None,
        epoch=-1,
    ) -> TrainingMetrics:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_energy = 0.0
        total_loss = 0.0
        total_firing_rate = 0.0

        all_results = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.float().to(self.device)
                target = target.to(self.device)

                (spikes, mem), results = self.chip.run(self.model, input_data=data)
                acc, loss = self.calculate_accuracy(spikes, target)

                results["accuracy"] = acc
                results["loss"] = loss
                results["epoch"] = epoch
                all_results.append(results)

                total_correct += acc * target.size(0)
                total_samples += target.size(0)
                total_energy += results["total_energy_nJ"]
                total_loss += loss
                total_firing_rate += spikes.mean().item()

        metrics = TrainingMetrics(
            accuracy=total_correct / total_samples,
            energy_usage=total_energy / len(test_loader),
            epoch=-1,  # Indicates evaluation
            loss=total_loss / len(test_loader),
            firing_rate=total_firing_rate / len(test_loader),
        )

        all_results = pd.DataFrame(all_results)

        if pbar is not None:
            desc = f"Test Acc: {metrics.accuracy:.3f}, Energy: {metrics.energy_usage / 1000:.2f} uJ"
            pbar.set_description(desc)

        return metrics, all_results

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        n_epochs: int,
    ):
        pbar = tqdm(range(n_epochs), desc="Training: ", leave=False)
        for epoch in pbar:
            self.train_epoch(train_loader, epoch=epoch, n_epochs=n_epochs, pbar=pbar)
            metrics, pd_results = self.evaluate(test_loader, epoch=epoch, pbar=pbar)
            self.chip_results.append(pd_results)

    @property
    def pd_results(self) -> pd.DataFrame:
        if len(self.chip_results) == 0:
            return pd.DataFrame()
        else:
            results = pd.concat(self.chip_results)
            for k, v in self.config.items():
                results[k] = [v] * len(results)
            return results

    @property
    def device(self) -> str:
        return next(self.model.parameters()).device

    @property
    def pareto_tradeoff(self) -> pd.DataFrame:
        best_epoch_mean = (
            self.pd_results.groupby("epoch")
            .mean()
            .sort_values(by="accuracy", ascending=False)
            .iloc[0]
        )
        return calculate_pareto_score(
            best_epoch_mean["accuracy"], best_epoch_mean["total_energy_nJ"]
        )
