# SDR‑JAX

# JAX‑based implementations of Sufficient Dimension Reduction (SDR) algorithms

***
EXPERIMENTAL – The library is under active development, has only a small test suite, 
and is not intended for production workloads yet. 
Use it for research, prototyping, or learning purposes only. 
Contributions, Bug Reports, Feature Requests are all welcome.
Extensive use of LLMs (recommend [Lumo](https://lumo.proton.me/)) is used throughout and encouraged in contributions. 
***

# Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Implemented Methods](#implemented-methods)
- [Contributing and Testing](#contributing-and-testing)
- [Code Quality Principles](#code-quality-principles)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

# Overview

**SDR‑JAX** provides fast, GPU/TPU‑compatible implementations of 
*sufficient dimension reduction* ([SDR](https://en.wikipedia.org/wiki/Sufficient_dimension_reduction))
techniques using JAX. These methods aim to find low‑dimensional transformations
of the predictor that retain all information about the response variable(s).

Key goals:
- Performance: JIT‑compiled kernels and vectorised operations for large‑scale data.
- Research‑ready: References to original papers and links to further reading.

# Installation

See requirements.txt (to be added) for the required Python packages. 

GPU/TPU acceleration – Ensure the appropriate CUDA/cuDNN or TPU runtime is installed 
before installing JAX. See the official JAX installation
guide: https://github.com/google/jax#installation.

# Implemented Methods
- Sliced Inverse Regression (SIR) – Li, K‑C. (1991). Sliced Inverse Regression for 
Dimension Reduction.
- Sliced Average Variance Estimation (SAVE) – Cook, R. D., & Weisberg, S. (1991). 
Discussion of “Sliced Inverse Regression”.

# Contributing and Testing

We welcome contributions! Please follow these steps:
- Fork the repository.
- Create a new branch (git checkout -b feature/my‑feature).
- Add or modify code. 
- Write unit tests using pytest (see tests/ directory).
- Run the test suite and linting checks:
  - pytest                # run tests
  - flake8 src/ tests/    # style enforcement
  - ensure the documentation builds with sphinx
  - Open a Pull Request describing the change.

# Code Quality Principles

- Style – Enforced with flake8 (PEP8 compliance).
- Documentation – Generated via Sphinx (make html).
- JIT compilation – All heavy numeric kernels are wrapped with jax.jit
- Static typing – Type hints throughout the codebase, enforced with Pydantic.

# Documentation

The documentation is built with Sphinx and hosted on Read the Docs (link to be added).
Run locally:
- cd docs
- make html
- Open _build/html/index.html in a browser.

# Acknowledgements

The privacy‑first AI tool [Lumo](https://lumo.proton.me/) – developed by Proton AG – has 
been used extensively to develop, test, and refine the JAX implementations in this repository.

We would like to thank all our contributors.

# License

This project is released under the *TBD* License. See the LICENSE file (to be added) for details.