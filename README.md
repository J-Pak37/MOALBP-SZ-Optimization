# Scalability Analysis of Multi-Objective Metaheuristics for Assembly Line Balancing under Soft Zoning Constraints

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
[cite_start]This repository contains the source code, modified benchmark datasets, and experimental results for the research paper: **"Scalability Analysis of Multi-Objective Metaheuristics for Assembly Line Balancing under Soft Zoning Constraints (MOALBP-SZ)"**[cite: 1442, 1460]. 

[cite_start]Traditional exact solvers frequently suffer from the exponential growth of the search space, leading to severe computational deadlocks in large-scale industrial scenarios[cite: 1461, 1503, 1504]. [cite_start]To overcome this limitation, this project formulates a Mixed-Integer Linear Programming (MILP) model and comprehensively evaluates the scalability of six algorithmic approaches[cite: 1462]:
1. Exact Method (IBM ILOG CPLEX)
2. Pure Variable Neighborhood Search (VNS)
3. Hybrid VNS (integrated with a CP solver)
4. Non-dominated Sorting Genetic Algorithm II (NSGA-II)
5. Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
6. Multi-Objective Variable Neighborhood Search (MO-VNS) - *Proposed Method*

[cite_start]The primary objectives are to simultaneously minimize the number of workstations ($Z_1$) and minimize the total operational costs ($Z_2$), which strictly incorporates heavy penalty costs for any spatial zoning rule violations[cite: 1460, 1548].

## Repository Structure

The project is organized as follows:

```text
MOALBP-SZ-Optimization/
│
├── data/                      # Modified benchmark datasets
│   ├── Group_A/               # Small instances (n < 20)
│   ├── Group_B/               # Medium instances (20 <= n <= 50)
│   └── Group_C/               # Large-scale instances (n > 50)
│
├── src/                       # Source code for all algorithms
│   ├── exact_method/          # CPLEX-based MILP implementation
│   ├── single_objective/      # Pure VNS and Hybrid VNS
│   ├── multi_objective/       # NSGA-II, MOEA/D, and MO-VNS implementations
│   └── utils/                 # Shared utilities (Decoding scheme, Pareto evaluation)
│
├── results/                   # Experimental logs and generated Pareto Front plots
└── README.md                  # Project documentation
