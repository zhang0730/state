from typing import Dict, List, Optional, Union, Literal
import functools
import torch
from torch.utils.data import Dataset, Subset
from data.data_modules.perturbation_tracker import PerturbationTracker
from data.utils.data_utils import split_perturbations_by_cell_fraction
import h5py
import numpy as np
from pathlib import Path
import logging

from data.mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)


class DatasetType:
    """An enum-like class"""
    REPLOGLE = "replogle"
    MCFALINE = "mcfaline"
    JIANG = "jiang"
    SCIPLEX = "sciplex"

def get_dataset_type(name: str) -> DatasetType:
    """Maps dataset name from config to DatasetType enum"""
    name = name.lower()
    if name == "replogle":
        return DatasetType.REPLOGLE
    elif name == "mcfaline":
        return DatasetType.MCFALINE 
    elif name == "jiang":
        return DatasetType.JIANG
    elif name == "sciplex":
        return DatasetType.SCIPLEX
    else:
        raise ValueError(f"Unknown dataset type: {name}")

class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from h5 files.

    Each instance handles a single dataset-cell_type combination. Therefore this class is responsible for
    serving a single dataset/cell_type pair. Future improvements will also allow for splitting on 
    the perturbation level.

    Currently there are three strategies for mapping basal cells to perturbed cells:
    - "batch": Basal cells are sampled from the same batch as the perturbed cell
    - "random": Basal cells are sampled randomly from the same cell type as the perturbed cell
    - "nearest": Basal cells are sampled from the nearest neighbors of the perturbed cell

    A control cell is always mapped to a perturbed cell within the same dataset and with the same cell type.
    """

    def __init__(
        self,
        name: str,
        h5_path: Union[str, Path],
        mapping_strategy: BaseMappingStrategy,
        pert_onehot_map: Optional[Dict[str, int]] = None,
        batch_onehot_map: Optional[Dict[str, int]] = None,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_uce", "X_pca"] = "X_uce",
        store_raw_expression: bool = False,
        random_state: int = 42,
        pert_tracker: Optional[PerturbationTracker] = None,
        should_yield_control_cells: bool = False,
        split_train_val_controls: bool = False,
        preload_data: bool = False,
        **kwargs,
    ):
        """
        Args:
            name: Name of the dataset
            h5_path: Path to the h5 file containing the dataset
            mapping_strategy: Strategy for mapping basal cells to perturbed cells, one of "batch", "random", "nearest"
            pert_onehot_map: Global mapping of perturbation names to one-hot encodings
            batch_onehot_map: Global mapping of batch names to one-hot encodings
            pert_col: Column in the h5 file containing perturbation information
            cell_type_key: Column in the h5 file containing cell type information
            batch_col: Column in the h5 file containing batch information
            control_pert: Name of the control perturbation
            embed_key: Key in the h5 file containing the expression data, one of "X" or "X_uce"
            random_state: Random seed for reproducibility
            pert_tracker: PerturbationTracker instance for tracking valid perturbations
            should_yield_control_cells: If True, control cells will be included in the dataset
        """
        super().__init__()
        self.name = name
        self.h5_path = Path(h5_path)
        self.mapping_strategy = mapping_strategy
        self.pert_onehot_map = pert_onehot_map
        self.batch_onehot_map = batch_onehot_map
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.store_raw_expression = store_raw_expression
        self.rng = np.random.RandomState(random_state)
        self.pert_tracker = pert_tracker
        self.should_yield_control_cells = should_yield_control_cells
        self.should_yield_controls = should_yield_control_cells
        self.split_train_val_controls = split_train_val_controls

        # Load file
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.pert_categories = self.h5_file[f'obs/{self.pert_col}/categories'][:].astype(str)
        self.cell_type_categories = self.h5_file[f'obs/{self.cell_type_key}/categories'][:].astype(str)

        try: # replogle treats batch as an int
            self.batch_is_categorical = False
            self.batch_categories = self.h5_file[f'obs/{self.batch_col}'][:].astype(int)
        except TypeError: # but some other datasets have it as a string (categorical)
            self.batch_is_categorical = True
            self.batch_categories = self.h5_file[f'obs/{self.batch_col}/categories'][:].astype(str)

        # Some cached info
        self.n_genes = self._get_num_genes()
        self.n_cells = self._get_num_cells()
        self.cell_type = str(self.cell_type_categories[0])  # single type per file

        # Identify which cells are controls vs. perturbed
        self.control_mask = (
            self.h5_file[f'obs/{self.pert_col}/codes'][:] ==
            np.where(self.pert_categories == self.control_pert)[0]
        )

        # TODO-Abhi: we should move this logic to MultiDatasetPerturbationDataModule
        if self.pert_tracker is not None:
            cell_type = str(self.cell_type_categories[0])  # single type per file
            dataset_name = name
            self.valid_perts = self.pert_tracker.track_celltype_perturbations(
                self.h5_file,
                dataset_name,
                cell_type
            )

        # We'll store the indices for each split. 
        self.split_perturbed_indices = {'train': set(), 'train_eval': set(), 'val': set(), 'test': set()}
        self.split_control_indices = {'train': set(), 'train_eval': set(), 'val': set(), 'test': set()}

        self.preloaded_data = {}
        if preload_data:
            logger.info(f"[{self.name}] Preloading all data into memory...")
            self._preload_all_data()

    def set_store_raw_expression(self, flag: bool):
        """
        Dynamically enable/disable whether this dataset yields raw gene expression.

        This is so that we don't load raw expression, which can be expensive, if we are only training
        latent models.
        """
        self.store_raw_expression = flag
        logger.info(f"[{self.name}] to yield raw gene expression: {flag}")

    def __len__(self) -> int:
        return self.n_cells

    def reset_mapping_strategy(self, new_strategy: BaseMappingStrategy, stage='train', **strategy_kwargs):
        """
        Re-run register_split_indices(...) for each known split in this dataset,
        using the new mapping strategy. This preserves the exact same indices
        (perturbed + control) but changes how basal controls will be sampled.
        """
        self.mapping_strategy = new_strategy(**strategy_kwargs)
        self.mapping_strategy.stage = stage
        for split_name in self.split_perturbed_indices:
            # gather perturbed + control as arrays
            pert_array = np.array(list(self.split_perturbed_indices[split_name]))
            ctrl_array = np.array(list(self.split_control_indices[split_name]))
            # call the new strategy’s register
            self.mapping_strategy.register_split_indices(
                self, split_name, pert_array, ctrl_array
            )
    
    def __getitem__(self, idx: int):
        """
        Get a control matching data from the dataset, given its index.

        This method will return a dictionary with the following keys:
        - 'X': the perturbed cell's expression data
        - 'X_gene': this is returned only if store_raw_expression is set
        - 'basal': the mapped control cell's expression data
        - 'pert': a one-hot encoding or a featurization of the perturbation type
        - 'pert_name': the name of the perturbation
        - 'cell_type': the cell type of the perturbed cell
        - 'gem_group': the batch of the perturbed cell
        """
        # Determine which split this cell belongs to (only needed if we do subset direct).
        split = self._find_split_for_idx(idx)
        
        if not self.should_yield_control_cells and self.control_mask[idx]:
            raise ValueError(f"Index {idx} is a control cell in {self.name}")

        # Get expressions (pseudobulked if using that strategy)  
        pert_expr, ctrl_expr = self.mapping_strategy.get_mapped_expressions(self, split, idx)
        
        # Rest of metadata...
        pert_code = self.h5_file[f"obs/{self.pert_col}/codes"][idx]
        pert_name = self.pert_categories[pert_code]
        pert_onehot = self.pert_onehot_map[pert_name]
        
        ct_code = self.h5_file[f"obs/{self.cell_type_key}/codes"][idx]
        cell_type = self.cell_type_categories[ct_code]
        batch = self.get_batch(idx)
        
        sample = {
            "X": pert_expr,
            "basal": ctrl_expr, 
            "pert": pert_onehot,
            "pert_name": pert_name,
            "cell_type": cell_type,
            "gem_group": batch,
        }
        
        if self.store_raw_expression:
            sample["X_gene"] = self.fetch_gene_expression(idx)
            
        return sample

    def get_batch(self, idx: int) -> torch.Tensor:
        """
        Get the batch information for a given cell index. Returns a scalar tensor.
        """
        assert self.batch_onehot_map is not None, "No batch onehot map, run setup."

        if self.batch_is_categorical:
            gem_code = self.h5_file[f"obs/{self.batch_col}/codes"][idx]
            batch_name = self.batch_categories[gem_code]
        else:
            batch_name = str(self.h5_file[f"obs/{self.batch_col}"][idx])

        batch = torch.argmax(self.batch_onehot_map[batch_name])
        return batch.item()

    def get_dim_for_obsm(self, key: str) -> int:
        """
        Get the feature dimensionality of obsm data with the specified key (e.g., 'X_uce').
        """
        return self.h5_file[f'obsm/{key}'].shape[1]

    # TODO-Abhi: can we move perturbed idx logic and control idx logic internally so these don't have to be passed in?
    def to_subset_dataset(
        self,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
        subsample_fraction: float = 1.0,
    ) -> Subset:
        """
        Creates a Subset of this dataset that includes only the specified perturbed_indices.
        If `self.should_yield_control_cells` flag is True, the Subset will also yield control cells.

        Args:
            split: Name of the split to create, one of 'train', 'val', 'test', or 'train_eval'
            perturbed_indices: Indices of perturbed cells to include
            control_indices: Indices of control cells to include
            subsample_fraction: Fraction of perturbed cells to include, the rest are ignored (default 1.0)
        """
        if np.any(self.control_mask[perturbed_indices]):
            raise ValueError("Trying to treat control cells as perturbed, but they are controls.")

        if subsample_fraction < 1.0:
            # randomly subsample the perturbed indices
            n_keep = int(len(perturbed_indices) * subsample_fraction)
            perturbed_indices = self.rng.choice(perturbed_indices, size=n_keep, replace=False)

        # sort them for stable ordering
        perturbed_indices = np.sort(perturbed_indices)
        control_indices = np.sort(control_indices)

        # Register them in the dataset
        self._register_split_indices(split, perturbed_indices, control_indices)

        # Return a Subset containing perturbed cells and optionally control cells
        if self.should_yield_control_cells:
            all_indices = np.concatenate([perturbed_indices, control_indices])
            return Subset(self, all_indices)
        else:
            return Subset(self, perturbed_indices)

    def prepare_training_splits(
        self,
        val_split: float = 0.10,
        rng: np.random.Generator = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val splits based on perturbation categories while 
        maintaining approximate size ratios. Control cells are shared between splits.
        
        Args:
            val_split: Fraction of data to use for validation
            rng: Random number generator for reproducibility
            
        Returns:
            Dict of splits containing perturbed and control indices
        """
        if rng is None:
            rng = np.random.default_rng(42)
            
        # Get all perturbations (excluding control)
        pert_codes = self.h5_file[f'obs/{self.pert_col}/codes'][:]
        pert_names = self.pert_categories[pert_codes]
        
        # Group cell indices by perturbation
        pert_groups = {}
        for pert in np.unique(pert_names):
            if pert == self.control_pert:
                continue
            pert_groups[pert] = np.where(pert_names == pert)[0]
        
        # Split while maintaining size proportions
        train_perts, val_perts = split_perturbations_by_cell_fraction(
            pert_groups,
            val_fraction=val_split,
            rng=rng,
        )
        
        # Gather indices for each split
        train_indices = np.concatenate([pert_groups[p] for p in train_perts])
        val_indices = np.concatenate([pert_groups[p] for p in val_perts])
        
        # 3) Control cells: either shared or split
        ctrl_indices = np.where(self.control_mask)[0]
        if not self.split_train_val_controls:
            # If we are NOT splitting controls, train+val both see the same controls
            train_ctrl = ctrl_indices
            val_ctrl   = ctrl_indices
        else:
            # If we DO split controls, we can do a straightforward proportion:
            rng.shuffle(ctrl_indices)
            n_val_ctrl = int(len(ctrl_indices) * val_split)
            val_ctrl   = ctrl_indices[:n_val_ctrl]
            train_ctrl = ctrl_indices[n_val_ctrl:]
        
        return {
            'train': {
                'perturbed': train_indices,
                'control': train_ctrl,
            },
            'val': {
                'perturbed': val_indices,
                'control': val_ctrl,
            }
        }

    def prepare_fewshot_splits(
        self,
        few_shot_percent: float = 0.3,
        val_split: float = 0.15,
        rng: np.random.Generator = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create train/val/test splits for few-shot learning, splitting on perturbations.
        
        Args:
            few_shot_percent: Fraction of data to use for train+val
            val_split: Fraction of train+val to use for validation
            rng: Random number generator for reproducibility
            
        Returns:
            Dict of splits containing perturbed and control indices
        """
        if rng is None:
            rng = np.random.default_rng(42)
            
        # Get all perturbations (excluding control)
        pert_codes = self.h5_file[f'obs/{self.pert_col}/codes'][:]
        pert_names = self.pert_categories[pert_codes]
        
        # Group indices by perturbation
        pert_groups = {}
        for pert in np.unique(pert_names):
            if pert == self.control_pert:
                continue
            pert_groups[pert] = np.where(pert_names == pert)[0]
        
        train_val_perts, test_perts = split_perturbations_by_cell_fraction(
            pert_groups,
            1-few_shot_percent,
            rng=rng,
        )
        
        # Then split train+val into train vs val
        train_val_pert_groups = {
            pert: pert_groups[pert] for pert in train_val_perts
        }
        train_perts, val_perts = split_perturbations_by_cell_fraction(
            train_val_pert_groups,
            val_split,
            rng=rng,
        )
        
        # Gather indices for each split
        train_indices = np.concatenate([pert_groups[p] for p in train_perts])
        val_indices = np.concatenate([pert_groups[p] for p in val_perts])  
        test_indices = np.concatenate([pert_groups[p] for p in test_perts])
        
        # 4) Handle control cells. By default, test has separate controls.
        ctrl_indices = np.where(self.control_mask)[0]
        n_test_ctrl = int(len(ctrl_indices) * (1 - few_shot_percent))
        train_val_ctrl = ctrl_indices[:-n_test_ctrl]   # for train+val
        test_ctrl      = ctrl_indices[-n_test_ctrl:]   # separate for test
        
        if not self.split_train_val_controls:
            # If not splitting controls, then train+val share the same control cells
            train_ctrl = train_val_ctrl
            val_ctrl   = train_val_ctrl
        else:
            # If we DO want to split train vs val controls as well
            rng.shuffle(train_val_ctrl)
            n_val_ctrl = int(len(train_val_ctrl) * val_split)
            val_ctrl   = train_val_ctrl[:n_val_ctrl]
            train_ctrl = train_val_ctrl[n_val_ctrl:]

        return {
            'train': {
                'perturbed': train_indices,
                'control': train_ctrl
            },
            'val': {
                'perturbed': val_indices,
                'control': val_ctrl
            },
            'test': {
                'perturbed': test_indices,
                'control': test_ctrl
            }
        }
    
    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        if hasattr(self, "preloaded_data") and 'X' in self.preloaded_data:
            return torch.tensor(self.preloaded_data['X'][idx], dtype=torch.float32)

        attrs = dict(self.h5_file['X'].attrs)
        if attrs['encoding-type'] == 'csr_matrix':
            indptr = self.h5_file["/X/indptr"]
            start_ptr = indptr[idx]
            end_ptr = indptr[idx + 1]
            sub_data = torch.tensor(
                self.h5_file["/X/data"][start_ptr:end_ptr],
                dtype=torch.float32
            )
            sub_indices = torch.tensor(
                self.h5_file["/X/indices"][start_ptr:end_ptr],
                dtype=torch.long
            )
            counts = torch.sparse_csr_tensor(
                torch.tensor([0], dtype=torch.long),
                sub_indices,
                sub_data,
                (1, self.n_genes),
            )
            data = counts.to_dense().squeeze()
        else:
            row_data = self.h5_file["/X"][idx]
            data = torch.tensor(row_data)
        return data
    
    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        if hasattr(self, "preloaded_data") and key in self.preloaded_data:
            return torch.tensor(self.preloaded_data[key][idx], dtype=torch.float32)
        row_data = self.h5_file[f"/obsm/{key}"][idx]
        return torch.tensor(row_data)

    def get_gene_names(self) -> List[str]:
        """
        Get the gene names, which are under adata.var.index, using h5.
        """
        try:
            genes = self.h5_file["var/gene_name"][:].astype(str).tolist()
        except:
            try:
                categories = self.h5_file["var/gene_name/categories"][:].astype(str)
                codes = self.h5_file["var/gene_name/codes"][:]
                genes = categories[codes].tolist()
            except:
                genes = self.h5_file["var/_index"][:].astype(str).tolist()

        return genes

    ##############################
    # Static methods
    ##############################
    @staticmethod
    def collate_fn(batch, transform=None):
        """
        Custom collate function that can apply transforms on batched data.

        The transform is bound to the dataset instance via a partial, hence why this 
        is a static method.
        """
        # First do normal collation
        batch_dict = {
            "X": torch.stack([item["X"] for item in batch]),
            "basal": torch.stack([item["basal"] for item in batch]),
            "pert": torch.stack([item["pert"] for item in batch]),
            "pert_name": [item["pert_name"] for item in batch],
            "cell_type": [item["cell_type"] for item in batch], 
            "gem_group": torch.tensor([item["gem_group"] for item in batch]),
        }

        # If the first sample has "X_gene", assume the entire batch does
        if "X_gene" in batch[0]:
            batch_dict["X_gene"] = torch.stack(
                [item["X_gene"] for item in batch]
            )

        # Apply transform if provided
        if transform is not None:
            batch_dict["X"] = transform.encode(batch_dict["X"]) 
            batch_dict["basal"] = transform.encode(batch_dict["basal"])

        return batch_dict


    ##############################
    # Utility methods
    ##############################
    def _register_split_indices(
        self,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray
    ):
        """
        Register which cell indices belong to the perturbed vs. control set for
        a given split. 
        
        These are passed to the mapping strategy to let it build its internal structures as needed.
        """
        if split not in self.split_perturbed_indices:
            raise ValueError(f"Invalid split {split}")

        # set them in the dataset
        self.split_perturbed_indices[split] = set(perturbed_indices)
        self.split_control_indices[split] = set(control_indices)

        # forward these to the mapping strategy
        self.mapping_strategy.register_split_indices(self, split, perturbed_indices, control_indices)

        logger.info(
            f"[{self.name}] Registered {split} split for {self.cell_type}: "
            f"{len(perturbed_indices)} perturbed, {len(control_indices)} controls."
        )

    def _find_split_for_idx(self, idx: int) -> Optional[str]:
        """Utility to find which split (train/val/test) this idx belongs to."""
        for s in self.split_perturbed_indices.keys():
            if idx in self.split_perturbed_indices[s] or idx in self.split_control_indices[s]:
                return s
        return None

    @functools.lru_cache
    def _get_num_genes(self) -> int:
        try:
            n_cols = self.h5_file["X"].shape[1]
        except:
            indices = self.h5_file["X/indices"][:]  # shape: (nnz,)
            n_cols = indices.max() + 1
        return n_cols

    @functools.lru_cache
    def _get_num_cells(self) -> int:
        try:
            n_rows = self.h5_file["X"].shape[0]
        except:
            indptr = self.h5_file["X/indptr"][:]    # shape: (n_rows+1,)
            n_rows = len(indptr) - 1
        return n_rows
    
    def __getstate__(self):
        """
        Return a dictionary of this dataset's state without the open h5 file object.
        """
        # Copy the object's dict
        state = self.__dict__.copy()
        # Remove the open file object if it exists
        if "h5_file" in state:
            # We'll also store whether it's currently open, so that we can re-open later if needed
            del state["h5_file"]
        return state

    def __setstate__(self, state):
        """
        Reconstruct the dataset after unpickling. Re-open the HDF5 file by path.
        """
        # TODO-Abhi: remove this before release
        self.__dict__.update(state)
        # This ensures that after we unpickle, we have a valid h5_file handle again
        self.h5_file = h5py.File(self.h5_path, "r")

    def _preload_all_data(self):
        """Preload all data into memory."""
        logger.info(f"[{self.name}] Preloading all data into memory...")
        
        # Load gene expression
        self.preloaded_data['X'] = torch.stack([
            self.fetch_gene_expression(i) for i in range(self.n_cells)
        ])
            
        # Load embeddings if used
        if self.embed_key:
            self.preloaded_data[self.embed_key] = torch.stack([
                self.fetch_obsm_expression(i, self.embed_key) for i in range(self.n_cells)
            ])
            
        logger.info(f"[{self.name}] Preload complete.")
