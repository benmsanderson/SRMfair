
# Stochastic SRM Modeling Framework

This repository contains a stochastic modeling framework that combines the **FaIR simple climate model** with an **adaptive and stochastic solar radiation modification (SRM) deployment module**, integrated with **economic damage functions**. The framework allows exploration of the interactions between SRM deployment, mitigation, and climate outcomes under uncertainty.

## Overview

The model simulates global mean temperature (GMT) trajectories under combined emissions and SRM pathways, accounting for uncertainties in climate response, SRM efficacy, operational risks (interruptions), and mitigation behavior (moral hazard). Damages are calculated based on both temperature levels and rates of change, under different discounting assumptions.

## Key Components

### 1. Climate Response Model: FaIR

The **FaIR (Finite Amplitude Impulse Response) simple climate model** is used to simulate GMT responses to emissions and radiative forcing. Uncertainty in climate response is sampled via ensembles of key parameters such as **transient climate response (TCR)**, **equilibrium climate sensitivity (ECS)**, and **carbon cycle feedbacks**.

### 2. Adaptive SRM Deployment: `adpt_fair`

The `adpt_fair` function implements an **adaptive SRM deployment strategy**, which dynamically adjusts SRM radiative forcing to maintain GMT below a target threshold (e.g., 2°C).

**Main Features of `adpt_fair`:**
- Iteratively adjusts optimal SRM forcing (`F_SRM_opt`) to maintain temperature targets.
- Incorporates SRM efficacy (`Effic`) to represent real-world deployment limitations.
- Includes stochastic interruptions (`simulate_failure`) based on annual failure probability (`pfail`) and average outage length (`aol`).
- Modulates emissions during SRM deployment via a moral hazard parameter (`mhaz`), which slows mitigation if SRM is active.
- Operates over a user-defined SRM deployment period (`Start` to `End`), adapting forcing annually.

### 3. SRM Configuration Parameters

SRM pathways (SRMPs) are defined using a dataframe (`df`) with the following key parameters:
- `Start`: Start year for SRM deployment.
- `End`: End year for SRM deployment.
- `threshold`: GMT target to be maintained (°C).
- `Effic`: SRM efficacy (fraction of radiative forcing achieved).
- `pfail`: Annual probability of SRM failure.
- `aol`: Mean duration of SRM failure (years).
- `mhaz`: Moral hazard strength (mitigation slowdown fraction).
- `sint`: Step size for adaptive SRM forcing adjustments.

### 4. Damage Functions

Damages are calculated as functions of both **temperature levels and rates of change**. Two types of damage function configurations are supported:
- **Conservative**: Level-dependent only, quadratic in temperature anomaly.
- **Ethical/Rate-sensitive**: Includes terms for rate of warming and applies lower discount rates to emphasize long-term risks.

**Damage function parameters** include:
- `a`: Coefficient on squared temperature anomaly.
- `b`: Coefficient on squared rate of temperature change.
- `r`: Discount rate for time integration.

## Workflow and Usage

### Notebook: `FAIR_SRM.ipynb`

The main analysis workflow is implemented in `FAIR_SRM.ipynb`, which performs the following steps:
1. Load and define SSP emissions pathways and SRM parameter sets (SRMPs).
2. Sample FaIR parameter ensembles for probabilistic climate response.
3. For each SSP–SRMP combination:
    - Run `adpt_fair` to generate temperature and SRM forcing trajectories.
    - Apply stochastic SRM failure using `simulate_failure`.
    - Adjust emissions for moral hazard effects.
    - Compute damages using chosen damage functions.
4. Aggregate results for analysis and visualization (e.g., GMT trajectories, SRM forcing, damages).

### Key Functions in `functions.py`
- `adpt_fair`: Adaptive SRM deployment calculation.
- `simulate_failure`: Models SRM interruptions.
- `compute_damages`: Calculates damages based on temperature trajectories and damage function parameters.

## Model Extensions and Next Steps

Future work may extend this framework to include:
- Alternative damage formulations (e.g., tipping point risks).
- Policy optimization studies combining SRM and mitigation.
- Coupling to regional or sectoral impact models.

## References
- Smith et al. (2018). *Geoscientific Model Development*, 11(6), 2273–2297.
- Glanemann et al. (2020). *Environmental Research Letters*, 15(10).
- Lemoine & Traeger (2014). *American Economic Journal: Economic Policy*, 6(1), 137–166.
- Stern (2007). *The Economics of Climate Change: The Stern Review.* Cambridge University Press.
- MacMartin et al. (2014, 2018). *Philosophical Transactions of the Royal Society A*.

## License
[Specify license here, e.g., MIT License or Creative Commons]
