# RoughVolatilityWorkshop

Python translation of the R materials from [QuantMinds Rough Volatility Workshop Lectures](https://github.com/jgatheral/RoughVolatilityWorkshop), with minor updates.

## Repository Structure

**Jupyter Notebooks**:

- [QM2024_1_Econometrics.ipynb](./QM2024_1_Econometrics.ipynb):
  - Shape of the volatlity surface

  - Scaling of implied volatility smiles

  - Monofractal scaling of realized variance

  - Estimation of $H$

  - Realized variance forecasting

- [QM2024_2_Rough_volatility_models.ipynb](./QM2024_2_Rough_volatility_models.ipynb):
  - The forward variance curve

  - Change of measure

  - The rough Bergomi model

  - The rough Heston model

  - The quadratic rough Heston model

  - Financial meaning of parameters

- [QM2024_3_Affine_models.ipynb](./QM2024_3_Affine_models.ipynb):
  - The microstructural foundation of affine forward variance models

  - Characteristic function methods
    - Option pricing
    - The ATM skew
    - The skew-stickiness ratio

  - Diamonds and the forest expansion

  - Moment computations

- [QM2024_4_Computations.ipynb](./QM2024_4_Numerical_methods.ipynb):
  - Rational approximation of rough Heston

  - Smile plotting and parameter sensitivities

  - The HQE scheme

**Python files**:

- [fwd_var_curve.py](./fwd_var_curve.py): Forward variance curve construction and smoothing
- [heston.py](./heston.py): Classical Heston model functions
- [pade.py](./pade.py): Padé/rational approximation coefficients for rough Heston
- [rough_bergomi.py](./rough_bergomi.py): Implementation of the rough Bergomi model
- [rough_heston.py](./rough_heston.py): Implementation of the rough Heston model
- [utils.py](./utils.py): Utility functions for the workshop
- [utils_plot.py](./utils_plot.py): Functions for plotting implied volatility curves and Monte Carlo results
- [var_gamma_swaps.py](./var_gamma_swaps.py): Variance and gamma swap calculations

## Python Installation Guide

### Option 1: Standard Virtual Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fbourgey/RoughVolatilityWorkshop.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd RoughVolatilityWorkshop
   ```

3. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

4. **Activate the environment:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies:**

   ```bash
   pip install .
   ```

6. **Launch Jupyter Lab (optional):**

   ```bash
   jupyter lab
   ```

---

### Option 2: Using `uv` (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed, setup is simpler and faster.  
After cloning the repository (steps 1–2 above), run:

```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies.
