# Growth Models Implementation 
#### Casus C Hanze Hogeschool Groningen 2024-2025

## Overview:
This Python script provides a comprehensive implementation of various mathematical growth models and plot the output vs the actual data
### Features:
#### Growth Models:
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
   - CombinedModel dV/dt = c * V^(2/3) - d * V

##### New StochasticGompertzGrowth arikel:
https://iopscience.iop.org/article/10.1088/1742-6596/1366/1/012018/

##### CombinedModel:
The model is used to describe the growth of a population in a limited environment. The model is a combination of the Von Bertalanffy growth model and the logistic growth model. The Von Bertalanffy growth model describes the growth of a population in an unlimited environment, while the logistic growth model describes the growth of a population in a limited environment.

#### Visualization examples:
In the included file(use_case.ipynb) there is some example data and visualization examples with it. It includes 3 different visualizations. One has all the models available in one plot for easy comparison. The other one an seperate plot for each model. The last one is a table with the AIC, BIC and AICc values for each model, it's in the order of ascending BIC value.

#### Example Data:
   - Can be found in the use_case.ipynb and the data folder.

### Requirements:
- Python 3.x
- Libraries:
  - matplotlib
  - numpy
  - random
  - pandas
  - scipy.optimize
  - math
  - skopt

### How to Run:
1. Ensure the required libraries are installed. Use `pip install matplotlib numpy pandas random scipy.optimize math skopt` to install them if necessary.
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

### Errors:
- Surface-Limited Growth, MendelsohnGrowth, MontrollGrowth: seems to not be working with the current implementation.
