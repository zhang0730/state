import numpy as np
from .mapping_strategies import BaseMappingStrategy

class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control(s) drawn from the entire set of
    controls in that split (same cell type).
    """
    def __init__(self, name="random", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples)
        self.split_control_pool = {
            'train': [],
            'train_eval': [],
            'val': [],
            'test': []
        }
    
    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray
    ):
        """
        Simply store all control indices in a list for each split.
        """
        self.split_control_pool[split] = control_indices

    def get_control_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_idx: int
    ) -> np.ndarray:
        if len(self.split_control_pool[split]) == 0:
            raise ValueError("No control cells found in RandomMappingStrategy.")
        return self.rng.choice(
            self.split_control_pool[split],
            size=self.n_basal_samples,
            replace=True
        )