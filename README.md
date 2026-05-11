# Code Repository for ICML 2026 Paper

**Title:** [Towards Understanding Generalization of Federated Adversarial Learning:
Perspective of Algorithmic Stability]

To facilitate reproducibility, we provide the complete codebase for our Federated Adversarial Learning (FAL) framework. This appendix outlines the repository structure, environment configuration, and core submodules.


# 1. Create and activate virtual environment
conda create -n fal_env python=3.8
conda activate fal_env

# 2. Install core PyTorch and dependency libraries
pip install torch torchvision numpy pandas scikit-learn
pip install matplotlib seaborn tqdm opencv-python

## Project Structure

```
├── FAL
│   ├── FAL_epsilon           # FAL with varying epsilon (privacy/attack)
│   ├── FAL_MNIST             # MNIST-based federated experiments
│   ├── FAL_SVHN              # SVHN-based federated experiments
│   ├── FAL_webspam           # Webspam dataset experiments
│   ├── FAL_Cifar10           # CIFAR-10 federated experiments
│   └── FAL_SUSY              # SUSY (HEP) federated experiments
├── FalME                     # Model Ensemble experiments
├── FalZO                     # Zero-Order optimization experiments
├── data                      # Dataset folder (place data here)
├── LICENSE                   # License
├── README.md                 # Main documentation
├── requirments.txt           # Python dependencies
```
## Quick Start

1. **Install dependencies:**

```bash
pip install -r requirments.txt
```

2. **Prepare dataset:**

Download or place datasets in the `data/` subfolders, e.g.,
- `data/mnist/`
- `data/svhn/`
- `data/webspam/`
- `data/cifar10/`
- `data/susy/`

3. **Run experiments:**

Navigate to a submodule and run the main script. For example:

```bash
cd FAL/FAL_epsilon
python fal_main_epsilon.py
```

Other modules follow a similar structure. Check each submodule's README or the `options.py` for available arguments and configurations.

---

## Module Overview

### 1. FAL/FAL_epsilon
Federated adversarial learning with varying epsilon (privacy or attack strength). Includes attacks, training, and visualization.

- `fal_main_epsilon.py`: Main entry point for FAL-epsilon experiments
- `attacks.py`: Attack methods
- `models.py`, `models2.py`: Model architectures
- `options.py`: Experiment parameters
- `sampling.py`, `update.py`, `utils.py`: FL core logic
- `plot_*.py`: Visualization scripts
- `pkl*.py`, `delete.py`: Helper scripts for managing result files

#### How to Run

```bash
cd FAL/FAL_epsilon
python fal_main_epsilon.py --option1 value1 ...
```

See `options.py` for all configurable parameters.

---

### 2. FAL/FAL_MNIST
Federated adversarial learning and ablation studies on the MNIST dataset.

- `FAL_delta_main.py`, `FAL_numusers_main.py`: Main scripts for delta and user experiments
- `models.py`: MNIST models
- `options.py`: Configurations
- `sampling.py`, `update.py`, `utils.py`: Core FL logic
- `plot*.py`: Visualization
- `FAL_lep_frac_src/`: Experiments on local epsilon privacy, user fractions

#### How to Run

```bash
cd FAL/FAL_MNIST
python FAL_delta_main.py --option1 value1 ...
```

---

### 3. FAL/FAL_SVHN
Federated adversarial learning on the SVHN dataset.

- `src/FAL_SVHN.py`: Main script
- `src/attack_generator.py`: Attack generation
- `src/models.py`: SVHN models
- `src/options.py`, `src/sampling.py`, `src/update.py`, `src/utils.py`: Core FL components
- `src/plot*.py`: Visualization
- `src/logger.py`: Logging utilities

#### How to Run

```bash
cd FAL/FAL_SVHN/src
python FAL_SVHN.py --option1 value1 ...
```

---

### 4. FAL/FAL_webspam
Federated adversarial learning on webspam (binary classification).

- `fal_main_webspam.py`, `fl_main_webspam.py`: Main FL and FAL scripts
- `attacks.py`, `models.py`, `options.py`: Core components
- `sampling.py`, `update.py`, `utils.py`: FL logic
- `plot.py`, `plot_results.py`: Visualization
- `pkl_json.py`, `delete.py`: Result management

#### How to Run

```bash
cd FAL/FAL_webspam
python fal_main_webspam.py --option1 value1 ...
```

---

### 5. FAL/FAL_Cifar10
Federated adversarial learning on CIFAR-10 (10-class image classification).

- Typical entry scripts: FAL/FL drivers (e.g., `fal_main_cifar10.py`, `fl_main_cifar10.py`)
- Core components usually include: `attacks.py`, `models.py`, `options.py`, `sampling.py`, `update.py`, `utils.py`
- Visualization/result helpers: `plot*.py`, `pkl_*.py`, `delete.py` (if provided)

#### How to Run

```bash
cd FAL/FAL_Cifar10
python fal_main_cifar10.py --option1 value1 ...
```

---

### 6. FAL/FAL_SUSY
Federated adversarial learning on SUSY (binary classification for high-energy physics).

- Typical entry scripts: FAL/FL drivers (e.g., `fal_main_susy.py`, `fl_main_susy.py`)
- Core components usually include: `attacks.py`, `models.py`, `options.py`, `sampling.py`, `update.py`, `utils.py`

#### How to Run

```bash
cd FAL/FAL_SUSY
python fal_main_susy.py --option1 value1 ...
```
### 7. data/
Place datasets here. Subfolders for MNIST, SVHN, webspam, etc.

---

## Dependencies

See `requirments.txt` for details. Main dependencies:

```
numpy
torch
torchvision
matplotlib
scikit-learn
pandas
tqdm
seaborn
opencv-python
```

---

## License

See LICENSE file for details.

---


