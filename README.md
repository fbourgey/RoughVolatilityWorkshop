# RoughVolatilityWorkshop

Python translation of the R materials from [QuantMinds Rough Volatility Workshop Lectures](https://github.com/jgatheral/RoughVolatilityWorkshop), with minor updates.

## Python Installation Guide

### Option 1: Standard Virtual Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jgatheral/QuadraticRoughHeston.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd QuadraticRoughHeston
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
