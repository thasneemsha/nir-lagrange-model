# NIR Glucose Prediction using Lagrange Interpolation

## Project Overview

This project applies **Lagrange interpolation** to estimate glucose concentration from high-dimensional Near-Infrared (NIR) spectral data.

The goal is to model smooth functional relationships between wavelength intensity patterns and glucose concentration.

---

# Dataset Overview

## Near-Infrared (NIR) Glucose Spectroscopy Dataset

This project uses a Near-Infrared (NIR) spectroscopy dataset designed for non-invasive glucose monitoring research.

A spectrometer emits infrared light across multiple wavelengths and records absorption patterns, forming biochemical fingerprints of the sample.

---

## Dataset Composition

| Dataset | Samples | Features |
|---|---:|---:|
| Calibration Set | 273 | 4,209 |
| Validation Set | 120 | 4,209 |

Each sample contains:
- 9 metadata variables
- 4,200 spectral wavelength features

---

## Metadata Variables

| Variable | Type | Range | Description |
|---|---|---|---|
| Glucose (mM) | Continuous | 0–50 | Target variable |
| Lactate (mM) | Continuous | 0–46 | Interferent |
| Acetaminophen | Continuous | — | Interferent |
| Caffeine | Continuous | — | Interferent |
| Ethanol | Continuous | 0–69 | Interferent |
| Temperature (°C) | Continuous | ~37–38 | Condition |
| Kuvette | Integer | — | Container ID |
| Day | Integer | — | Experiment day |
| Run | Integer | — | Mixture run |

---

## Spectral Features

- Wavelength range: **400 nm → 2499.5 nm**
- Step size: **0.5 nm**
- Each value:  
  `A = -log(I/I₀)`

---

## Glucose Distribution

| Dataset | Unique Values | Range |
|---|---:|---:|
| Calibration | 24 | 0–50 mM |
| Validation | 20 | 0–50 mM |

---

## Clinical Categories

| Level | Meaning | Samples |
|---|---|---:|
| <4 mM | Hypoglycemia | 63 |
| 4–7 mM | Normal | 21 |
| >7 mM | Hyperglycemia | 189 |

---

# Technique Overview

Lagrange interpolation estimates glucose values by constructing a polynomial passing through known calibration points.

It is used here as a baseline smooth curve estimator for spectral-response relationships.

---

# Mathematical Model

Lagrange polynomial:

\[
L(x)=\sum y_i L_i(x)
\]

\[
L_i(x)=\prod_{j \ne i} \frac{x-x_j}{x_i-x_j}
\]

---

# Pipeline

```
Raw Spectra → Feature Selection → Interpolation Points
→ Lagrange Model → Prediction → Evaluation
```

---

# Results

Outputs include:
- Predicted glucose curves
- Error comparison plots
- Validation performance metrics

---

# Repository Structure

```
src/lagrange_prediction.py
docs/lagrange_theory.md
results/
data/
```

---

# Notes

- Works best on smooth calibration curves
- Sensitive to noise in high-dimensional spectral data
