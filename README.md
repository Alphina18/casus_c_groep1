README: Growth Models Implementation 
Opdracht: Casus C Hanze Hogeschool Groningen 2024-2025

Overview:
This Python script provides a comprehensive implementation of various mathematical growth models and plot the output vs the actuel data
Features:
1. Growth Models:
   - Linear Growth: dV/dt = c
   - Exponential Gry_actualowth: dV/dt = c * V
   - Mendelsohn Growth: dV/dt = c * V^d
   - Exponentially Decaying Growth: dV/dt = c * (V_max - V)
   - Logistic Growth: dV/dt = c * V * (V_max - V)
   - Montroll Growth: dV/dt = c * V * (V_dmax - V)
   - Allee Effect Growth: dV/dt = c * (V - V_min) * (V_max - V)
   - Linearly Limited Growth: dV/dt = c * V / (V + d)
   - Surface-Limited Growth: dV/dt = c * V * (V + d)^(1/3)
   - Von Bertalanffy Growth: dV/dt = c * V^(2/3) - d * V
   - Gompertz Growth: dV/dt = c * V * ln(V_max / V)\
   - StochasticGompertzGrowth  ΔV=c⋅Vln( V_max/V )Δt+σ⋅V Δt⋅ϵ(t)

New StochasticGompertzGrowth arikel:
https://iopscience.iop.org/article/10.1088/1742-6596/1366/1/012018/

2. Visualization:
   - The script plots all models on a single figure to compare their behavior over time.
   - Each plot includes a title, labeled axes, gridlines, and a legend for clarity.

3. Example Data:
   - Includes example time (`ts`) and value (`Vs`) data for real-world comparisons.

Requirements:
- Python 3.x
- Libraries:
  - matplotlib
  - numpy
  - random

How to Run:
1. Ensure the required libraries are installed. Use `pip install matplotlib numpy pandas`
2. Run the script in a Python environment:
   ```bash
   python code.py
Give param as number how to get the data
	1: Load data from a CSV file
	2: Enter data manually
	3: Generate random data
Give param as string 'euler', 'heun', 'runge_kutta' for wich method to use
Give as a string 'direct', 'random_search' to wich method to use to find the best params.
Give as a string 'AIC', 'AICC', 'BIC' wich calculation need to be addressed.

Errors:
- the "direct" and "runge_kutta" method are hard tested and mokey proof
- 'AIC', 'AICC', 'BIC' calculations are correct
