import torch
import numpy as np
from typing import Dict, Optional
from models import SNNModel

# DO NOT CHANGE THESE PARAMETERS
ENERGY_NORMALIZATION_FACTOR = 10_000
PARETO_ALPHA = 0.5


class NeuromorphicChip:
    def __init__(self):
        """
        Memory and energy parameters for the neuromorphic chip
        ⚡ DO NOT CHANGE THESE PARAMETERS. THESE ARE THE CONSTRAINTS YOU NEED TO WORK WITH ⚡
        """
        self.MAX_NEURONS = 1024
        self.MAX_SYNAPSES = self.MAX_NEURONS * 64
        self.MEMORY_PER_NEURON = 32  # bytes
        self.MEMORY_PER_SYNAPSE = 4  # bytes
        self.TOTAL_MEMORY = (
            self.MAX_SYNAPSES * self.MEMORY_PER_SYNAPSE
            + self.MAX_NEURONS * self.MEMORY_PER_NEURON
        )

        self.ENERGY_PER_NEURON_UPDATE = 1e-1  # nJ
        self.ENERGY_PER_SYNAPSE_EVENT = 5e-4  # nJ

        self.mapped_snn = None

    def calculate_memory_usage(self, snn: SNNModel) -> int:
        """
        Calculate total memory usage for the given SNN
        TODO: Implement this method, using the total number of neurons and synapses of the SNN.
        /!\ : You need to implement the properties n_neurons and n_synapses in the SNN class first.

        """
        raise NotImplementedError("Memory usage not implemented")

    def map(self, snn: SNNModel) -> bool:
        """
        Map the given SNN to the chip. This method should check if the SNN fits on the chip
        and map it to the chip if it does, by setting the self.mapped_snn attribute. If it doesn't fit, raise a MemoryError.
        TODO: Implement this method, using the total number of neurons and synapses
        """
        self.mapped_snn = snn
        raise NotImplementedError("Mapping not implemented")

    def run(
        self, snn: Optional[SNNModel] = None, input_data: torch.Tensor = None
    ) -> Dict:
        """
        Run the mapped SNN and return performance metrics. The steps are the following:
        1/ Run the SNN simulation
        TODO: Implement the rest of the method.
        2/ Compute the total number of spikes and the spike rate
        3/ Compute the total energy consumed by the SNN
        4/ Return the results in a dictionary
        """

        if snn is not None:
            # Map the SNN to the chip and check if it fits
            self.map(snn)

        # Run the actual network simulation. We don't need to compute gradients for this.
        with torch.no_grad():
            spk_rec, mem_rec = self.mapped_snn(input_data)

        # Get network recordings for all layers.
        recordings = self.mapped_snn.recordings

        # Calculate spike metrics
        total_spikes = None  # TODO: Calculate total number of spikes
        spike_rate = None  # TODO: Calculate spike rate

        # Calculate energy metrics
        # TODO: Get the total number of neuron updates. This should not depend on the recordings.
        total_neuron_updates = None

        # TODO: Get the total number of synapse events. This should depend on the recordings.
        # To get the total number of synapse events, we need to sum the number of
        # spikes x the number of synapses for each layer. For a dense cinnectivity this is straightforward.
        # For a sparse connectivity, we need to sum the number of non-zero weights in the synapse matrix.
        total_synapse_events = None

        # TODO: Calculate energy metrics. To do so, use the chip energy parameters.
        energy_neurons = None
        energy_synapses = None
        total_energy = None

        # Return the results in a dictionary
        sim_results = {
            "total_energy_nJ": total_energy,
            "memory_usage_bytes": self.calculate_memory_usage(self.mapped_snn),
            "neuron_updates": total_neuron_updates,
            "synapse_events": total_synapse_events,
            "spike_rate": spike_rate,
            "total_spikes": total_spikes,
        }

        raise NotImplementedError("Simulation results not implemented")

        return (spk_rec, mem_rec), sim_results


def calculate_pareto_score(accuracy: float, energy_nj: float) -> float:
    """
    Calculate Pareto trade-off score between accuracy and energy.

    Args:
        accuracy: Classification accuracy (0 to 1)
        energy_nj: Energy consumption in nanojoules

    Returns:
        Combined score (higher is better)
    """
    # Accuracy term (higher is better)
    accuracy_term = PARETO_ALPHA * accuracy

    # Energy efficiency term (lower energy is better, so we invert it)
    # Normalized to 0-1 range using ENERGY_NORMALIZATION_FACTOR
    energy_efficiency = (
        ENERGY_NORMALIZATION_FACTOR - energy_nj
    ) / ENERGY_NORMALIZATION_FACTOR
    energy_term = (1 - PARETO_ALPHA) * energy_efficiency

    return accuracy_term + energy_term
