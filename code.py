import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define each of the growth models (same as before)

def linear_growth(t, V, c):
    """Linear Growth Model: dV/dt = c"""
    return c

def exponential_growth_derivative(t, V, c):
    """Exponential Growth Model: dV/dt = c * V"""
    return c * V

def mendelsohn_growth(t, V, c, d):
    """Mendelsohn Growth Model: dV/dt = c * V^d"""
    return c * V**d

def exponential_decay_growth(t, V, c, V_max):
    """Exponentially Decaying Growth Model: dV/dt = c * (V_max - V)"""
    return c * (V_max - V)

def logistic_growth(t, V, c, V_max):
    """Logistic Growth Model: dV/dt = c * V * (V_max - V)"""
    return c * V * (V_max - V)

def montroll_growth(t, V, c, V_dmax):
    """Montroll Growth Model: dV/dt = c * V * (V_dmax - V)"""
    return c * V * (V_dmax - V)

def allee_effect_growth(t, V, c, V_min, V_max):
    """Allee Effect Growth Model: dV/dt = c * (V - V_min) * (V_max - V)"""
    return c * (V - V_min) * (V_max - V)

def linearly_limited_growth(t, V, c, d):
    """Linearly Limited Growth Model: dV/dt = c * V / (V + d)"""
    return c * V / (V + d)

def surface_limited_growth(t, V, c, d):
    """Surface-Limited Growth Model: dV/dt = c * V * (V + d)^(1/3)"""
    low = 1e-10
    if V + d <= 0:  
        return 0
    return c * V * (V + d + low)**(1/3)

def von_bertalanffy_growth(t, V, c, d):
    """Von Bertalanffy Growth Model: dV/dt = c * V^(2/3) - d * V"""
    return c * V**(2/3) - d * V

def gompertz_growth(t, V, c, V_max):
    """Gompertz Growth Model: dV/dt = c * V * ln(V_max / V)"""
    return c * V * np.log(V_max / V)

def time(seq):
    """Generate continuous time array for the number of steps in a given sequence"""
    return np.linspace(0, seq - 1, seq)

def solve_model(model_func, t_span, y0, t_eval, *params):
    solution = solve_ivp(model_func, t_span, y0, t_eval=t_eval, args=params, method='RK45')
    
    # Debugging: Check if the solution is correct
    if solution.success:
        print(f"Model {model_func.__name__}: Success, time range: {solution.t[0]} - {solution.t[-1]}")
    else:
        print(f"Model {model_func.__name__}: Failed")
    
    return solution.t, solution.y[0]  

def make_plot(models, t, labels):
    """Plot all models on a single figure."""
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
    plt.plot(t, model, label=labels[i]) 
    plt.title('Growth Models Comparison')
    plt.xlabel('Time (t)')
    plt.ylabel('Value (V)')
    plt.grid(True)
    plt.legend() 
    plt.show()

def run_model():
    ts = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
    Vs = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]
    t = time(len(ts))  
    
    v = 0.1  
    c = 0.0001 
    V_max = 1
    V_dmax = 0.4
    V_min = 0.001
    d = 0.5  


    
    y0 = [v]

    models = [
        (linear_growth, c),
        (exponential_growth_derivative, c),
        (mendelsohn_growth, c, d),
        (logistic_growth, c, V_max),
        (montroll_growth, c, V_dmax),
        (allee_effect_growth, c, V_min, V_max),
        (linearly_limited_growth, c, d),
        (surface_limited_growth, c, d),
        (von_bertalanffy_growth, c, d),
        (gompertz_growth, c, V_max)
    ]
    
    t_span = (0, 100)  # From t=0 to t=100
    t_eval = np.linspace(0, 100, 1000)  #consistent t
    
    model_data = []
    labels = []
	# alle plot bij elkaar
    for model_func, *params in models:
        t_solution, y_solution = solve_model(model_func, t_span, y0, t_eval, *params)
        
        y_solution_interp = np.interp(t_eval, t_solution, y_solution)
        
        model_data.append(y_solution_interp)  
        labels.append(model_func.__name__)

    make_plot(model_data, t_eval, labels)

run_model()
