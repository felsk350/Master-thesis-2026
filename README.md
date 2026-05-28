# Master-thesis-2025

# Align and Compare Sheets

This project processes 3D scan data of sheet metal parts, aligns multiple scans within defined groups, computes mean surfaces, and performs deviation analysis against both experimental and simulated reference plates.

It supports:
- Within-group alignment and averaging
- Cross-group comparisons
- Comparison against simulated finite element plates
- Full deviation statistics and visualization outputs

---

## Overview

The pipeline performs the following steps:

### 1. Per group processing
For each group of scanned sheets:
- Detects 5 geometric landmarks (4 corners + crease vertex)
- Aligns all sheets to a reference scan using Kabsch alignment
- Builds a common grid representation of all sheets
- Computes:
  - Mean surface
  - Standard deviation map
  - Per-sheet deviation from mean
- Identifies the sheet closest to the group mean

### 2. Cross-group comparison
- Compares mean surfaces between groups
- Compares the most representative (closest-to-mean) sheets

### 3. Simulation comparison
- Aligns simulated plates to experimental coordinate systems
- Compares:
  - Each sheet vs simulation
  - Mean sheet vs simulation

---

## Dependencies

Install required Python packages:

```bash
pip install numpy scipy matplotlib plyfile

# Curvature Analysis

This script computes **Gaussian curvature** for aligned 3D sheet metal scans, mean surfaces, and simulated reference plates. It builds on outputs from `align_and_compare.py`.

## Features

- Gaussian curvature computation from height fields
- Per-sheet curvature analysis (aligned scans)
- Mean sheet curvature per group
- Curvature analysis of simulated plates
- Curvature difference vs simulation
- Cross-group curvature comparison
- Outputs CSV, PLY, and visual maps

## Requirements

```bash
pip install numpy scipy matplotlib plyfile
