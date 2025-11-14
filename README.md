# Parametric Curve Fitting

This project estimates the parameters **θ**, **M**, and **X** for a nonlinear curve that best fits the given dataset (`xy_data.csv`).

It uses `scipy.optimize.least_squares` to minimize the **L1 distance** between the observed data and the theoretical curve.

Run with:
```bash
python code.py
