from collections import defaultdict
import numpy as np
import logging
from typing import Dict, Set, Tuple, Optional
import h5py

class PerturbationTracker:
    """Tracks perturbation presence across datasets and cell types."""
    
    def __init__(self):
        # Structure: dataset -> cell_type -> set(perturbations)
        self.perturbations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        # Structure: dataset -> cell_type -> perturbation -> count
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
    def track_celltype_perturbations(self, 
                                   h5_file: h5py.File,
                                   dataset_name: str,
                                   cell_type: str,
                                   pert_col: str = 'gene',
                                   min_count: int = 1) -> Set[str]:
        """
        Track perturbations present in a specific cell type within a dataset.
        
        Args:
            h5_file: Open h5py file containing data for one cell type
            dataset_name: Name of the dataset
            cell_type: Name of the cell type
            pert_col: Column containing perturbation labels
            min_count: Minimum number of cells required for a perturbation
            
        Returns:
            Set of valid perturbations for this cell type
        """
        if not dataset_name or not cell_type:
            raise ValueError("Dataset name and cell type must be non-empty strings")

        # Get perturbation categories and codes
        try:
            pert_categories = h5_file[f'obs/{pert_col}/categories'][:].astype(str)
            pert_codes = h5_file[f'obs/{pert_col}/codes'][:]
        except KeyError as e:
            raise KeyError(f"Could not find perturbation data in h5 file: {e}")
        
        # Count occurrences 
        unique_codes, counts = np.unique(pert_codes, return_counts=True)
        
        valid_perts = set()
        for code, count in zip(unique_codes, counts):
            pert = pert_categories[code]
            self.counts[dataset_name][cell_type][pert] = count
            if count >= min_count:
                valid_perts.add(pert)
                self.perturbations[dataset_name][cell_type].add(pert)
                
        return valid_perts
    
    def get_dataset_perturbations(self, dataset: str) -> Set[str]:
        """Get all perturbations present in any cell type in the dataset."""
        return set().union(*(
            perts for cell_type, perts in self.perturbations[dataset].items()
        ))
    
    def compute_shared_perturbations(self, eval_dataset: str) -> Set[str]:
        """
        Find perturbations shared between eval dataset and at least one other dataset.
        Considers a perturbation present in a dataset if it appears in any cell type.
        """
        if eval_dataset not in self.perturbations:
            raise KeyError(f"Dataset {eval_dataset} not found in tracker")

        eval_perts = self.get_dataset_perturbations(eval_dataset)
        other_datasets = set(self.perturbations.keys()) - {eval_dataset}

        if not other_datasets:
            logging.warning("No other datasets found for comparison") 
            return set()
        
        # Union of perturbations across all other datasets
        other_perts = set().union(*(
            self.get_dataset_perturbations(d) for d in other_datasets
        ))
        
        return eval_perts & other_perts
    
    def log_overlap_statistics(self, eval_dataset: str):
        """Log detailed statistics about perturbation overlap."""
        if eval_dataset not in self.perturbations:
            raise KeyError(f"Dataset {eval_dataset} not found in tracker")

        eval_perts = self.get_dataset_perturbations(eval_dataset)
        if not eval_perts:
            logging.warning(f"No perturbations found in dataset {eval_dataset}")
            return
        other_datasets = set(self.perturbations.keys()) - {eval_dataset}
        shared = self.compute_shared_perturbations(eval_dataset)
        
        logging.info(f"\nPerturbation overlap statistics for {eval_dataset}:")
        logging.info(f"Total perturbations in {eval_dataset}: {len(eval_perts)}")
        logging.info(f"Shared with other datasets: {len(shared)}")
        
        # Per-dataset overlap
        for other in other_datasets:
            other_perts = self.get_dataset_perturbations(other)
            overlap = eval_perts & other_perts
            logging.info(f"Overlap with {other}: {len(overlap)} perturbations")
            
        # Per-cell type details
        logging.info("\nPer cell type breakdown:")
        for cell_type in self.perturbations[eval_dataset]:
            cell_perts = self.perturbations[eval_dataset][cell_type]
            cell_shared = cell_perts & shared
            logging.info(f"{cell_type}: {len(cell_perts)} total, {len(cell_shared)} shared")
