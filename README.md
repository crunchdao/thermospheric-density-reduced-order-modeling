# ThermosphericDensity-ReducedOrderModeling

<div align="center">

[![Build status](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/workflows/build/badge.svg?branch=master&event=push)](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/actions?query=workflow%3Abuild)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/releases)
[![License](https://img.shields.io/github/license/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling)](https://github.com/pPoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/454101625.svg)](https://zenodo.org/badge/latestdoi/454101625)

Python package for `Thermospheric Density Reduced Order Modeling` created with https://github.com/TezRomacH/python-package-template


</div>

## Description

In the context of the commercial activities performed in Low-Earth Orbit, a region of space of a few hundred kilometers of altitude, because the space traffic is increasing, it is important for us to obtain a model of the density field of the atmosphere, as the motion of the satellites in this orbital regime is strongly influenced by atmospheric drag, which is a function of the atmospheric density. 

While such models, based on first principles, already exist, they are complex and require a lot of computations; at the same time, more empirical models are less accurate. 

The trade-off proposed in this work, called reduced-order modeling, enables us to obtain a compressed representation of the density field, which can be used to construct predictive models, to perform uncertainty quantification and estimate the position of spacecraft in the future taking into account our knowledge of the environment and our availability of observation data. 

We here focus on non-linear methods, to perform the compression, using Machine Learning Methods. In particular, the use of Neural Networks is compared with the use of Support Vector Machine Methods. Interestingly, for the datasets investigated, the latter technique is not only much more efficient, but also more accurate.

## Clone repository with Large Files (LFS)

Please install [git-lfs](https://git-lfs.github.com/) and use this command to clone our repository:

```
git lfs clone https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling.git
```

## Installation

[`Makefile`](https://github.com/PoincareTrajectories/ThermosphericDensity-ReducedOrderModeling/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks could be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

<details>
<summary>4. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>


## Related packages

- [SSMLearn](https://github.com/haller-group/SSMLearn)
- [Codpy](https://github.com/JohnLeM/codpy_alpha)
- [libROM](https://github.com/LLNL/libROM)


## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)
