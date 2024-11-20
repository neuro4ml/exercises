import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils


class RecordingSequential(nn.Sequential):
    """
    A Sequential container that records intermediate activations (spikes and membrane potentials)
    from all spiking layers during forward pass.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.reset_recordings()

    def reset_recordings(self):
        """Reset all spike and membrane potential recordings"""
        self._recordings = {
            "spikes": {},
            "membrane": {},
        }

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with recording of intermediate activations.

        Args:
            input: Input tensor

        Returns:
            tuple: (output_spikes, output_membrane, recordings)
        """
        # Reset hidden states at the start of each forward pass
        self.reset_recordings()

        x = input
        current_idx = 0

        # Process each layer while recording spiking layers
        for layer in self:
            if isinstance(
                layer, (snn.Leaky, snn.Synaptic)
            ):  # Record only spiking layers
                out = layer(x)
                if isinstance(out, tuple):
                    spk, *mem = out
                    mem = mem[-1]
                else:
                    spk, mem = out, None

                self._recordings["spikes"][current_idx] = spk
                self._recordings["membrane"][current_idx] = mem
                x = spk
                current_idx += 1
            else:
                x = layer(x)

        # Return the final layer's outputs
        return (
            self._recordings["spikes"][current_idx - 1],
            self._recordings["membrane"][current_idx - 1],
            self._recordings,
        )


class SNNModel(nn.Module):
    """
    A simplified Spiking Neural Network using snnTorch.
    You need to implement:
    1/ property n_neurons
    2/ property n_synapses
    """

    def __init__(
        self,
        n_in: int = 128,
        n_hidden: int = 128,
        n_out: int = 10,
        beta: float = 0.95,  # decay rate
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.layers = RecordingSequential(
            nn.Linear(self.n_in, self.n_hidden),
            snn.Leaky(beta=beta, init_hidden=True, output=True),
            nn.Linear(self.n_hidden, self.n_out),
            snn.Leaky(beta=beta, init_hidden=True, output=True),
        )

        self.n_timesteps = 100

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run network simulation for input x.
        Args:
            x: Input tensor of shape (time_steps, batch_size, input_size)
        Returns:
            tuple: (spike_recording, membrane_recording)
        """

        assert x.shape[0] == self.n_timesteps, str(
            f"Input tensor must have the correct number of time steps, shape is {x.shape} but should be batch x time x input_size"
        )
        # Initialize hidden states
        utils.reset(self.layers)

        # Record spikes for each time step
        spk_rec = []
        mem_rec = []
        self._recordings = {
            "spikes": {},
            "membrane": {},
        }

        for step, x_t in enumerate(x):
            spk, *mem, recordings = self.layers(x_t)
            mem = mem[-1]
            spk_rec.append(spk)
            mem_rec.append(mem)
            for rec in recordings:
                for k, v in recordings[rec].items():
                    self._recordings[rec].setdefault(k, []).append(v)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    @property
    def recordings(self):
        return {
            rec_name: {
                idx: torch.stack(recs) if (recs[0] is not None) else recs
                for idx, recs in self._recordings[rec_name].items()
            }
            for rec_name in self._recordings
        }

    @property
    def n_neurons(self) -> int:
        """
        TODO: Calculate total number of neurons in the network
        Hint: Use out_features of linear layers or use the dimensions that we used in the initialization
        """
        raise NotImplementedError("Number of neurons not implemented")

    @property
    def n_synapses(self) -> int:
        """
        TODO: Calculate total number of active synapses in the network
        Hint: Count non-zero weights in linear layers (access weights with layer.weight)
        Hint: you can use torch.count_nonzero(...), but remember to copy back to CPU with .cpu().data.item()
        Optional: Use weight masks to create sparse connectivity in the network, to reduce this number !
        """
        raise NotImplementedError("Number of synapses not implemented")

    def __repr__(self):
        return f"SNNModel(n_neurons={self.n_neurons}, n_synapses={self.n_synapses})"

    def to(self, device: str):
        self.layers.to(device)
        return self
