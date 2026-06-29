# GAL2DM_ASIM_VNET

V-Net-based 3D convolutional neural network implementation for dark matter density field reconstruction from galaxy observables using the A-SIM simulation.

## Features

- V-Net / 3D U-Net-style reconstruction model
- Conditional reconstruction from galaxy number density and peculiar velocity fields
- Efficient voxel-level 3D dark matter density prediction
- Training and prediction pipelines
- Evaluation scripts for voxel-level and cosmological statistics
- Support for best-parameter experiments, base-sweep studies, and random-seed analysis

## Repository Structure

```text
src/        # Model architecture and utilities
scripts/    # Training and prediction scripts
eval/       # Evaluation and analysis scripts
etc/        # Configuration files
debug/      # Debugging utilities
logs/       # Log files, ignored by git
results/    # Outputs/checkpoints, ignored by git
```

## Requirements

- Python 3.10+
- PyTorch
- CUDA
- NumPy
- h5py
- tqdm

## Notes

Large files such as datasets, model checkpoints, prediction outputs, and logs are not tracked in this repository.

## Citation

If you use this repository in your research, please cite the corresponding publication (to be added).