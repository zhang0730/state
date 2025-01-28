import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Iterator
from torch.utils.data import Sampler


class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring all elements in a batch share same cell type and perturbation.
    Control cells are treated as their own perturbation type.

    This sampler works with MetadataConcatDataset which is a wrapper around multiple
    PerturbationDataset subsets. It maintains global indices while correctly accessing
    the underlying h5 files using the proper index mapping.

    Args:
        dataset: MetadataConcatDataset containing multiple dataset subsets
        batch_size: Number of samples per batch
        drop_last: Whether to drop the last batch if it's smaller than batch_size
    """

    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False):
        if hasattr(dataset, "data_source"):
            self.dataset = (
                dataset.data_source
            )  # lightning sometimes tries to pass in dataset as a sequential sampler here...
        else:
            self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by (cell_type, perturbation), accounting for dataset offsets
        self.grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        offset = 0
        for subset in self.dataset.datasets:  # These are the individual dataset subsets
            base_dataset = subset.dataset  # This is the PerturbationDataset
            cell_type = base_dataset.cell_type_categories[0]

            # Go through all indices in this subset
            for i in range(len(subset)):
                global_idx = offset + i  # Index in the concatenated dataset
                base_idx = subset.indices[i]  # Actual index in the base h5 file

                # Get perturbation name using correct base index
                pert_code = base_dataset.h5_file[f"obs/{base_dataset.pert_col}/codes"][base_idx]
                pert_name = base_dataset.pert_categories[pert_code]

                # Group both control and perturbed cells
                self.grouped_indices[(cell_type, pert_name)].append(global_idx)

            offset += len(subset)

        # Calculate total number of batches
        self.num_batches = sum(
            len(indices) // self.batch_size + (0 if self.drop_last else 1) for indices in self.grouped_indices.values()
        )

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yields batches of indices where all cells in a batch share the same
        cell type and perturbation.

        Returns:
            Iterator yielding lists of indices for each batch
        """
        # Shuffle indices within each group
        grouped_indices_lists = {key: indices.copy() for key, indices in self.grouped_indices.items()}
        for indices in grouped_indices_lists.values():
            np.random.shuffle(indices)

        # Yield batches from each group
        for indices in grouped_indices_lists.values():
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                yield batch_indices

    def __len__(self) -> int:
        """Returns the total number of batches."""
        return self.num_batches
