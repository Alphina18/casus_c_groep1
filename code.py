import numpy as np
import matplotlib.pyplot as plt
from random import random
import csv

# Base class for GrowthModel
class GrowthModel:
    def __init__(self, solver='euler', optimizer='direct', evaluation='AIC'):
        self.solver = solver
        self.optimizer = optimizer
        self.evaluation = evaluation  # AIC, BIC, AICC
        self.params = {}

    required_params = set()

    def growth_rate(self, V, t=None):
        """To be implemented by subclasses."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def euler_method(self, t):
        steps = 100
        V = self.params["V0"]  # Initial condition
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            dVdt = self.growth_rate(V)
            V += dVdt * dt
        return V

    def heun_method(self, t):
        steps = 100
        V = self.params["V0"]  # Initial condition
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            k1 = self.growth_rate(V)
            print(k1)
            k2 = self.growth_rate(V + dt * k1)
            V += dt * 0.5 * (k1 + k2)
        return V

    def runge_kutta_method(self, t):
        steps = 100
        V = self.params["V0"]
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  
            k1 = self.growth_rate(V)
            k2 = self.growth_rate(V + 0.5 * dt * k1)
            k3 = self.growth_rate(V + 0.5 * dt * k2)
            k4 = self.growth_rate(V + dt * k3)
            V += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return V

    def solve(self, t):
        """Solve using the selected method."""
        if self.solver == 'euler':
            return self.euler_method(t)
        elif self.solver == 'heun':
            return self.heun_method(t)
        elif self.solver == 'runge_kutta':
            return self.runge_kutta_method(t)
        else:
            raise ValueError("Unknown solver method")

    def mean_squared_error(self, data_ts, data_ys, **params):
        N = len(data_ts)
        total = 0.0
        for i in range(N):
            V_pred = self.solve(data_ts[i])  # Predicted value from the solver method
            error = data_ys[i] - V_pred
            total += error ** 2
        return total / N

    def fit_data(self, data_ts, data_ys):
        self.vdata = data_ys
        self.tdata = data_ts
        if self.optimizer == 'direct':
            params = self.direct_search(data_ts, data_ys)
        elif self.optimizer == 'random_search':
            params = self.random_search(data_ts, data_ys)
        self.params = params

    def random_search(self, data_ts, data_ys):
        # Initialize parameters (this could be changed depending on the model)
        params = {key: 0.0 for key in self.required_params}  # Initial guess for parameters
        mse = self.mean_squared_error(data_ts, data_ys, **params)
        
        tries = 0
        while tries < 1000:
            tries += 1
            new_params = {key: val + (random() - 0.5) for key, val in params.items()}
            new_mse = self.mean_squared_error(data_ts, data_ys, **new_params)
            
            # If the new MSE is better, keep the parameters
            if new_mse < mse:
                params = new_params
                mse = new_mse
                tries = 0  # Reset tries if we find a better set of parameters
        print(f"Optimized parameters (after {tries} tries):")
        for key, val in params.items():
            print(f"* {key:>2s} = {val:9.6f}")
        return params

    def direct_search(self, data_ts, data_ys):
        params = {param: 1.0 for param in self.required_params}
        self.params = params
        deltas = {key: 1.0 for key in params}
        mse = self.mean_squared_error(data_ts, data_ys, **params)

        while min(abs(codedelta) for codedelta in deltas.values()) > 1e-8:
            for key in params:
                new_params = params.copy()
                new_params[key] = params[key] + deltas[key]
                self.params = new_params
                new_mse = self.mean_squared_error(data_ts, data_ys, **new_params)
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= 1.2
                    continue
                new_params[key] = params[key] - deltas[key]
                self.params = new_params
                new_mse = self.mean_squared_error(data_ts, data_ys, **new_params)
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= -1.0
                    continue
                deltas[key] *= 0.5
        return params

    def eval_aic(self):
        k = len(self.params)
        n = len(self.vdata)
        return n * np.log(self.mean_squared_error(self.tdata, self.vdata, **self.params)) + 2 * k

    def eval_aicc(self):
        k = len(self.params)
        n = len(self.vdata)
        return self.eval_aic() + (2 * k + (k + 1)) / (n - k - 1)

    def eval_bic(self):
        k = len(self.params)
        n = len(self.vdata)
        return n * np.log(self.mean_squared_error(self.tdata, self.vdata, **self.params)) + np.log(n) * k

    def evaluate(self):
        """Evaluates the model based on the selected metric."""
        if self.evaluation == 'AIC':
            return self.eval_aic()
        elif self.evaluation == 'AICC':
            return self.eval_aicc()
        elif self.evaluation == 'BIC':
            return self.eval_bic()
        else:
            raise ValueError("Unknown evaluation metric")

# Models
class LinearGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"]

class ExponentialGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V

class MendelsohnGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        # Implement overflow protection
        MAX_V = 1e6  # Set a limit for the volume value to avoid overflow
        V = min(V, MAX_V)  # Cap V to a safe value
        return self.params["c"] * (V ** self.params["d"])

class ExponentialDecayGrowth(GrowthModel):
    required_params = {"c", "Vmax", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (self.params["Vmax"] - V)

class LogisticGrowth(GrowthModel):
    required_params = {"c", "V0", "Vmax"}

    def growth_rate(self, V):
        return self.params["c"] * V * (self.params["Vmax"] - V)

class MontrollGrowth(GrowthModel):
    required_params = {"c", "d", "Vmax", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V * (self.params["Vmax"] ** self.params["d"] - V ** self.params["d"])

class AlleeEffectGrowth(GrowthModel):
    required_params = {"c", "d", "Vmax", "Vmin", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V - self.params["Vmin"]) * (self.params["Vmax"] - V)

class LinearLimitedGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}
    def growth_rate(self, V):
        calc = V + self.params["d"]
        if V <= 0:
            V = 1e6
        if calc == 0:
            calc = 1e6
        return self.params["c"] * (V / (calc))

class SurfaceLimitedGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        calc = V + self.params["d"]
        if V <= 0:
            V = 1e6
        if calc == 0:
            calc = 1e6
        return self.params["c"] * (V / ((V + self.params["d"]) ** (1 / 3)))

class VonBertalanffyGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V ** (2 / 3)) - self.params["d"] * V

class GompertzGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V * np.log(1.0 / V)

class StochasticGompertzGrowth(GrowthModel):
    required_params = {"a", "b", "V0", "sigma", "Vmax"}

    def growth_rate(self, V, t=None):
        """Stochastic Gompertz growth rate with Wiener process."""
        if t is None:
            t = 0
        a = self.params["a"]
        b = self.params["b"]
        sigma = self.params["sigma"]
        Vmax = self.params["Vmax"]
        
        # Wiener process with random noise
        noise = sigma * np.random.normal(0, 1)
        
        # Gompertz model with stochastic perturbation
        return a * V * np.log(1.0 / V) + noise

import numpy as np
import matplotlib.pyplot as plt
from random import random
import csv

# Data Normalization Function
def normalize_data(time_data, volume_data):
    """Normalize time and volume data to range [0, 1]."""
    time_min, time_max = np.min(time_data), np.max(time_data)
    volume_min, volume_max = np.min(volume_data), np.max(volume_data)
    
    time_data_normalized = (time_data - time_min) / (time_max - time_min)
    volume_data_normalized = (volume_data - volume_min) / (volume_max - volume_min)
    
    return time_data_normalized, volume_data_normalized

# Function to load data from a CSV file
def load_data_from_csv(filename):
    time_data = []
    volume_data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            time_data.append(float(row[0]))
            volume_data.append(float(row[1]))
    
    # Normalize the data
    time_data = np.array(time_data)
    volume_data = np.array(volume_data)
    return normalize_data(time_data, volume_data)

# Function to generate data and normalize
def generate_random_data(num_points=100, time_range=(0, 10), volume_range=(1, 10)):
    t_actual = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
    y_actual = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]

    # Scaling data
    scale_factor_t = 1.5 / max(t_actual)
    t_actual_scaled = [t * scale_factor_t for t in t_actual]
    y_actual_scaled = [y / max(y_actual) for y in y_actual]
    
    time_data = np.array(t_actual_scaled)
    volume_data = np.array(y_actual_scaled)

    return normalize_data(time_data, volume_data)

# Main function to fit models and plot results
def main():
    print("Choose the data source:")
    print("1: Load data from a CSV file")
    print("2: Enter data manually")
    print("3: Generate random data")
    data_source = input("Enter the number of your choice (1, 2, or 3): ").strip()

    if data_source == '1':
        print("Enter the file path of the data (e.g., 'data.csv'):")
        file_path = input().strip()
        time_data, volume_data = load_data_from_csv(file_path)
    elif data_source == '2':
        print("Enter time and volume data manually, separated by commas (type 'done' when finished):")
        time_data = []
        volume_data = []
        while True:
            data_input = input()
            if data_input.lower() == 'done':
                break
            t, v = map(float, data_input.split(','))
            time_data.append(t)
            volume_data.append(v)
        time_data = np.array(time_data)
        volume_data = np.array(volume_data)
        time_data, volume_data = normalize_data(time_data, volume_data)
    elif data_source == '3':
        time_data, volume_data = generate_random_data()  # Default settings
    else:
        print("Invalid choice!")
        return

    print("Enter the solver method ('euler', 'heun', 'runge_kutta'):")
    solver = input().strip()

    print("Enter the optimizer method ('direct', 'random_search'):")
    optimizer = input().strip()

    print("Enter the evaluation metric ('AIC', 'AICC', 'BIC'):")
    evaluation = input().strip()

    # Fit data to each model
    models = {
    "LinearGrowth": LinearGrowth,
    "ExponentialGrowth": ExponentialGrowth,
    "MendelsohnGrowth": MendelsohnGrowth,
    "ExponentialDecayGrowth": ExponentialDecayGrowth,
    "LogisticGrowth": LogisticGrowth,
    "MontrollGrowth": MontrollGrowth,
    "AlleeEffectGrowth": AlleeEffectGrowth,
    "LinearLimitedGrowth": LinearLimitedGrowth,
    "SurfaceLimitedGrowth": SurfaceLimitedGrowth,
    "VonBertalanffyGrowth": VonBertalanffyGrowth,
    "GompertzGrowth": GompertzGrowth,
    "StochasticGompertzGrowth": StochasticGompertzGrowth
    }
    plt.figure(figsize=(10, 6))
    plt.scatter(time_data, volume_data, color='black', label='Data', zorder=5)
    plt.xlabel('Normalized Time')
    plt.ylabel('Normalized Volume')
    plt.title('Growth Models Comparison')

    for model_name, model_class in models.items():
        print(f"\nFitting model: {model_name}")
        model = model_class(solver=solver, optimizer=optimizer, evaluation=evaluation)
        model.fit_data(time_data, volume_data)
        print(f"Parameters for {model_name}: {model.params}")
        print(f"Evaluation Metric ({evaluation}): {model.evaluate()}")

        # Plot 
        predicted_values = [model.solve(t) for t in time_data]
        plt.plot(time_data, predicted_values, label=model_name)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

