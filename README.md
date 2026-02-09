# CRUISE: Curriculum-Structured Reinforcement Learning via Iterative Self-Play

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of **CRUISE**, a learning framework for stabilizing **Independent Multi-Agent Reinforcement Learning** in competitive, non-stationary, continuous-control environments using **structured curriculum learning and iterative self-play**.

The framework is evaluated on a high-fidelity **multi-drone racing** benchmark, which serves as a challenging testbed for decentralized multi-agent learning under collision constraints and adversarial interactions.

This code accompanies the paper:  
**Stabilizing Independent Multi-Agent Reinforcement Learning via Curriculum-Based Iterative Self-Play**  
*Currently under review.*

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [License](#license)

---

## Overview

CRUISE (Curriculum-structured Reinforcement Learning via Iterative Self-Play) is a training framework designed to **stabilize fully decentralized reinforcement learning** in competitive multi-agent environments.

Unlike centralized training approaches, CRUISE operates strictly in the **Independent Learning (IL)** paradigm. Learning stability is achieved through:
- A **manually structured task curriculum** that progressively increases task difficulty, and
- A lightweight **iterative self-play mechanism** that mitigates non-stationarity during training.

The provided implementation demonstrates CRUISE on a competitive multi-drone racing scenario with realistic quadrotor dynamics. However, the framework is designed to be reusable for other competitive continuous-control domains.

---

## Example: Multi-Drone Trajectories

Below is an example visualization of four independently trained agents racing on a Figure-Eight track (`figs/FigEight_track.png`), illustrating emergent coordination and collision avoidance without centralized control.

![Four drone trajectories on a Fig-8 track](figs/FigEight_track.png)

---

## Installation

### Requirements
- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12

### Step 1: Create and activate the environment
```sh
conda create -n cruise python=3.12
conda activate cruise
```

### Step 2: Install dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Training
To train CRUISE using the structured curriculum and self-play mechanism:
```sh
python curriculum_train.py
```

### Evaluation
To evaluate pretrained models and reproduce results reported in the paper:
```sh
python test.py
```

### Pretrained Models
Pretrained policies corresponding to the experiments reported in the paper are included in this repository.
These models can be directly evaluated using `test.py` for benchmarking and analysis.

## Citation
The associated paper is currently under review.
If you use this code, please cite the paper once it becomes publicly available.
A BibTeX entry will be added to this repository upon publication.

## License
This repository is released under the MIT License.