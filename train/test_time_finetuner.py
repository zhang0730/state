from typing import Optional, Tuple
import torch
import logging
from torch.utils.data import DataLoader, Subset
from lightning import LightningModule, Trainer
from models.base import PerturbationModel
from data.data_modules.multidataset_dataloader import MetadataConcatDataset

logger = logging.getLogger(__name__)

class TestTimeFineTuner:
    """
    Handles test-time fine-tuning of a model on control cells from the test set.
    
    Currently only fine-tunes the main model (not any decoders) using the same 
    loss function, learning rate, etc. from training time.
    
    Args:
        model: The PerturbationModel to fine-tune
        test_loader: DataLoader containing test data (will be filtered to controls)
        control_pert: Name of the control perturbation
        num_epochs: Number of epochs to fine-tune
        device: Device to use for training
    """
    def __init__(
        self,
        model: PerturbationModel,
        test_loader: DataLoader,
        control_pert: str = "non-targeting",
        num_epochs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model
        self.test_loader = test_loader
        self.control_pert = control_pert
        self.num_epochs = num_epochs
        self.device = device
        
        # Extract original training hyperparameters from the model
        self.learning_rate = model.lr
        self.loss_fn = model.loss_fn
        
        # Create control-only dataloader
        self.control_loader = self._create_control_dataloader()

    def _create_control_dataloader(self) -> DataLoader:
        """
        Create a new DataLoader containing only control cells from test set.
        Preserves the original batch size, num_workers, and collate_fn.
        """
        if not isinstance(self.test_loader.dataset, MetadataConcatDataset):
            raise ValueError("Expected test_loader.dataset to be MetadataConcatDataset")
            
        # Get indices of control cells
        control_indices = []
        for idx, batch in enumerate(self.test_loader.dataset):
            if batch['pert_name'] == self.control_pert:
                control_indices.append(idx)
                
        if not control_indices:
            raise ValueError(f"No control cells ({self.control_pert}) found in test set")
            
        # Create new subset with just control cells
        control_dataset = Subset(self.test_loader.dataset, control_indices)
        
        # Create new loader with same params as test_loader
        control_loader = DataLoader(
            control_dataset,
            batch_size=self.test_loader.batch_size,
            shuffle=True,  # shuffle during fine-tuning
            num_workers=self.test_loader.num_workers,
            collate_fn=self.test_loader.collate_fn,
        )
        
        logger.info(f"Created control-only loader with {len(control_indices)} cells")
        return control_loader

    def finetune(self) -> None:
        """
        Fine-tune the model on control cells for specified number of epochs.
        Uses same loss function and optimizer settings as training time.
        """
        logger.info(f"Starting test-time fine-tuning for {self.num_epochs} epochs...")
        
        # Store original training state
        training = self.model.training
        
        # Set up for training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.control_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass (using same logic as training)
                pred = self.model(batch)
                if self.model.output_space == "gene":
                    loss = self.loss_fn(pred, batch["X_gene"])
                else:
                    loss = self.loss_fn(pred, batch['X'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Restore original training state
        self.model.train(training)
        logger.info("Test-time fine-tuning complete.")