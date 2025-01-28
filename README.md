# PertBench Documentation 

## Overview
PertBench is a framework for training and evaluating models that predict cellular responses to genetic perturbations. It supports multiple model architectures, cell types, and perturbation scenarios.

## Installation

```bash
# Clone the repository
git clone https://github.com/ArcInstitute/pert-bench
cd pert-bench

# Install requirements - create conda env from vci_pertbench_environment.yml
conda env create -f vci_pertbench_environment.yml -n vci2
conda activate vci2
```

## Data Setup

PertBench uses a standardized data structure for datasets:

```
data_dir/
├── replogle/
│   ├── jurkat.h5
│   ├── k562.h5
│   ├── hepg2.h5
│   └── rpe1.h5
├── jiang/
│   ├── k562.h5
│   └── hepg2.h5
└── ...
```

Each H5 file must contain:
- `.obs['cell_type']`: Cell type labels
- `.obs['gene']`: Perturbation labels
- `.obs['gem_group']`: Batch information
- `.X`: Gene expression matrix
- `.obsm['X_uce']` (optional): UCE embeddings if using latent space models

One H5 file should contain data for exactly one cell type. The data_dir parameter should be 
provided in configs/data/multidataset.yaml, or as a command line override.

## Model Options

PertBench provides several model architectures:

1. **GlobalSimpleSum**: Basic baseline that computes global control means and perturbation offsets.
   ```python
   model=globalsimplesum
   ```

2. **SimpleSum**: Maps between control and perturbed cells using specified strategy.
   ```python
   model=simplesum
   ```

3. **EmbedSum**: Projects perturbations and cell states into a shared latent space.
   ```python
   model=embedsum
   ```

4. **NeuralOT**: Model using GPT2 backbone and a distributional loss function.
   ```python
   model=neuralot
   ```

## Training Configuration

### Input/Output Spaces

PertBench supports two types of input/output configurations:

1. **Gene Expression Space**:
```yaml
data.kwargs.embed_key=None # Set the inputs and regression targets to be gene expression space
data.kwargs.output_space=gene # Set the output of the model to be gene expression space
```

2. **Latent Space** (recommended):
```yaml
data.kwargs.embed_key=X_uce  # or X_scGPT, 
data.kwargs.output_space=latent # Set the inputs and regression targets to the specified latent space
```

### Mapping Strategies

Available strategies for mapping perturbed cells to control cells:

1. **nearest**: Maps to closest control cell (training only)
```yaml
data.kwargs.basal_mapping_strategy=nearest
data.kwargs.k_neighbors=10
```

2. **batch**: Maps within same experimental batch
```yaml
data.kwargs.basal_mapping_strategy=batch
```

3. **random**: Random control cell mapping
```yaml
data.kwargs.basal_mapping_strategy=random
```

4. **pseudobulk**: Average over local neighborhood
```yaml
data.kwargs.basal_mapping_strategy=pseudobulk
data.kwargs.neighborhood_fraction=0.5  # Fraction of population for averaging higher means more aggressive pseudobulking
```

5. **pseudo_nearest**: For inference using learned offsets
```yaml
data.kwargs.basal_mapping_strategy=pseudo_nearest
```

### Task Specification

Specify training and testing tasks using dataset and cell type combinations:

```yaml
# Train on all cell types in replogle except jurkat
data.kwargs.train_task=replogle
data.kwargs.test_task=replogle_jurkat:zeroshot

# Train on multiple datasets
data.kwargs.train_task=replogle,jiang_K562
data.kwargs.test_task=replogle_rpe1:zeroshot,jiang_HepG2:fewshot
```

For few-shot learning, specify the percentage of test data to use in training:
```yaml
data.kwargs.few_shot_percent=0.3  # Use 30% for training
```

## Example Training Commands

1. Zero-shot learning on jurkat:
```bash
python -m benchmark.train \
    data.kwargs.train_task=replogle,jiang \
    data.kwargs.test_task=jiang_K562:zeroshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=latent \
    data.kwargs.basal_mapping_strategy=pseudobulk \
    data.kwargs.neighborhood_fraction=0.5 \
    data.kwargs.split_train_val_controls=True \
    model=neuralot \
    output_dir=/path/to/output \
    name=jiang_K562_zeroshot
```

2. Few-shot learning on k562:
```bash
python -m benchmark.train \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task=replogle_k562:fewshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=latent \
    data.kwargs.basal_mapping_strategy=pseudobulk \
    data.kwargs.neighborhood_fraction=0.5 \
    model=neuralot \
    output_dir=/path/to/output \
    name=k562_fewshot
```

## Model Evaluation

The evaluation script allows testing different mapping strategies:

```bash
python -m benchmark.train.evaluate_model \
    --output_dir /path/to/experiment \
    --checkpoint last.ckpt \
    --map_type pseudobulk  # Can differ from training strategy
```

Here, `/path/to/experiment` should be `/path/to/output/name` where `/path/to/output` and `name` are from the training command. Evaluation automatically computes:
- MSE
- Pearson correlation
- Wasserstein distance
- MMD
- Cosine similarity
- DE gene overlap

## Best Practices

1. **Model Selection**:
   - Use latent space models with output_space="latent" for best performance

2. **Mapping Strategy**:
   - For training: pseudobulk with neighborhood_fraction=0.5 is recommended
   - For evaluation: use same strategy as training for fair comparison
   
3. **Data Organization**:
   - Keep one cell type per H5 file
   - Ensure consistent gene names across datasets
   - Include batch information in gem_group

4. **Validation**:
   - Use control cells from test set for test-time fine-tuning
   - Compare multiple mapping strategies during evaluation

## Common Issues

1. Missing control cells in test set:
   - Ensure control cells (with label "non-targeting") are present
   - Check gem_group assignments
   - Make sure each h5 file has a cell_type variable in obs

2. Memory issues:
   - Use data.kwargs.preload_data=True for faster training
   - Adjust batch_size based on available GPU memory

# Below this is legacy

# Training scRNA-seq perturbation predictors
Here we explore how to train supported perturbation predictors on the prepared datasets. To begin with, the training scripts are available under `benchmark.scripts` module. Each script reads a model-specific configuration file from `configs` directory and trains the model on the specified dataset. 

* `cell_type`: The target cell type which is using to test the model
* `cell_filter`: Cell filteration method to use for the dataset
* `map`: Strategy to use for mapping control cells to perturbed cells

## NeuralOT
The NeuralOT model is trained with an efficient multi-dataset dataloader. This enables us to specify different generalization tasks, e.g., cell type transfer in the zeroshot or few shot setting.

An example command given values for `${OUTPUT_DIR}`, `${NAME}`, `${MAP_TYPE}`, `${NEIGHBORS}`, and `${TAGS}` is:
```bash
python -m benchmark.scripts_lightning.train_lightning \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task=replogle_jurkat:zeroshot \
    output_dir=${OUTPUT_DIR} \
    name=${NAME} \
    data.kwargs.basal_mapping_strategy=${MAP_TYPE} \
    data.kwargs.k_neighbors=${NEIGHBORS} \
    wandb.tags="[${TAGS}]"
```

Here, `train_task` should be a comma separated list of datasets to use (from replogle, sciplex, feng, jiang, and/or mcfaline). The `test_task` further specifies which cell types from those training datasets should be excluded for test data, e.g. above, we hold out the jurkat cell types from the replogle dataset for testing. 

The `basal_mapping_strategy` can be one of `nearest`, `batch`, or `random`. This species how we map perturbed cells to control cells. The `k_neighbors` parameter further specifies the number of nearest neighbors to use for the `nearest` mapping strategy.

An example command to then evaluate this model is:
```bash
python -m benchmark.scripts_lightning.eval_lightning \
    --output_dir ${OUTPUT_DIR} \
    --name ${NAME} \
    --checkpoint ${MODEL_CKPT} \
    --map_type ${MAP_TYPE}
```
This tracks several metrics including the overlap in predicted vs true DE genes. The script will log these metrics to the same wandb run as the training script.

Here, `${OUTPUT_DIR}` and `${NAME}` should be the same as the training run that this is evaluating, and the checkpoint should be one of the checkpoints from that run (the last checkpoint will be `${OUTPUT_DIR}/${NAME}/version_0/checkpoints/last.ckpt`). Lastly, `map_type` can differ from the training run, but should be one of `nearest`, `batch`, or `random`. This determines how we sample perturbed and control cells for inference / evaluations.

## CPA
```bash
python -m benchmark.scripts.train_cpa cell_type=replogle_jurkat cell_filter=edist_train map=batch
```

## scVI
```bash
python -m benchmark.scripts.train_scvi cell_type=replogle_jurkat cell_filter=edist_train map=batch
```

## scGPT
NOTE: In order to successfully run the following command (specifically loading the original pre-trained weights without errors), you need to have `flash_attn==1.0.2` installed on your python environment.
```bash
python -m benchmark.scripts.train_scgpt cell_type=replogle_jurkat cell_filter=edist_train map=batch
```
