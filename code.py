import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class GrowthModelAnalyzer:
    def __init__(self, ts, Vs):
        self.ts = self.normalize_data(ts)
        self.Vs = self.normalize_data(Vs)
        self.models = {
            'linear_growth': self.linear_growth,
            'exponential_growth': self.exponential_growth_derivative,
            'mendelsohn_growth': self.mendelsohn_growth,
            'logistic_growth': self.logistic_growth,
            'montroll_growth': self.montroll_growth,
            'allee_effect_growth': self.allee_effect_growth,
            'linearly_limited_growth': self.linearly_limited_growth,
            'surface_limited_growth': self.surface_limited_growth,
            'von_bertalanffy_growth': self.von_bertalanffy_growth
        }
        self.initial_params = {
            'linear_growth': [0.1],
            'exponential_growth': [0.1],
            'mendelsohn_growth': [0.1, 0.1],
            'logistic_growth': [0.1, 1.0],
            'montroll_growth': [0.1, 1.0],
            'allee_effect_growth': [0.1, 0.1, 1.0],
            'linearly_limited_growth': [0.1, 0.1],
            'surface_limited_growth': [0.1, 0.1],
            'von_bertalanffy_growth': [0.1, 0.1]
        }

    @staticmethod
    def normalize_data(data):
        max_data = max(data)
        return [i / max_data if i != 0 else i for i in data]

    @staticmethod
    def linear_growth(t, V, c):
        return c

    @staticmethod
    def exponential_growth_derivative(t, V, c):
        return c * V

    @staticmethod
    def mendelsohn_growth(t, V, c, d):
        return c * V**d

    @staticmethod
    def logistic_growth(t, V, c, V_max):
        return c * V * (V_max - V)

    @staticmethod
    def montroll_growth(t, V, c, V_dmax):
        return c * V * (V_dmax - V)

    @staticmethod
    def allee_effect_growth(t, V, c, V_min, V_max):
        return c * (V - V_min) * (V_max - V)

    @staticmethod
    def linearly_limited_growth(t, V, c, d):
        return c * V / (V + d)

    @staticmethod
    def surface_limited_growth(t, V, c, d):
        epsilon = 1e-10
        if V + d <= 0:
            return 0
        return c * V * (V + d + epsilon)**(1/3)

    @staticmethod
    def von_bertalanffy_growth(t, V, c, d):
        if V <= 0:
            V = 1e-6
        return c * V**(2/3) - d * V

    @staticmethod
    def solve_ivp_model(model_func, t_span, y0, t_eval, *params):
        solution = solve_ivp(model_func, t_span, y0, t_eval=t_eval, args=params, method='RK45')
        return solution.t, solution.y[0]

    @staticmethod
    def mean_squared_error(params, model_func, ts, Vs):
        y0 = [Vs[0]]
        t_span = (ts[0], ts[-1])
        t_eval = ts
        t_solution, y_solution = GrowthModelAnalyzer.solve_ivp_model(model_func, t_span, y0, t_eval, *params)
        y_solution_interp = np.interp(ts, t_solution, y_solution)
        mse = np.mean((y_solution_interp - Vs) ** 2)
        return mse

    @staticmethod
    def calculate_aic_bic(mse, k, n):
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + np.log(n) * k
        return aic, bic

    @staticmethod
    def calculate_aicc(aic, k, n):
        return aic + (2 * k * (k + 1)) / (n - k - 1)

    def optimize_parameters(self, model_func, ts, Vs, initial_params):
        result = minimize(
            self.mean_squared_error, 
            initial_params, 
            args=(model_func, ts, Vs), 
            method='Nelder-Mead'
        )
        return result.x

    def run_analysis(self):
        n = len(self.ts)
        optimized_params_dict = {}
        aicc_dict = {}
        aic_bic_dict = {}

        for model_name, model_func in self.models.items():
            initial_params = self.initial_params[model_name]
            optimized_params = self.optimize_parameters(model_func, self.ts, self.Vs, initial_params)
            mse = self.mean_squared_error(optimized_params, model_func, self.ts, self.Vs)
            k = len(initial_params)
            aic, bic = self.calculate_aic_bic(mse, k, n)
            aicc = self.calculate_aicc(aic, k, n)

            optimized_params_dict[model_name] = optimized_params
            aicc_dict[model_name] = aicc
            aic_bic_dict[model_name] = (aic, bic)

            print(f"Model: {model_name}, AIC: {aic:.2f}, BIC: {bic:.2f}, AICc: {aicc:.2f}, Parameters: {optimized_params}")

        best_model = min(aicc_dict, key=aicc_dict.get)
        print(f"\nBest Model: {best_model} with AICc: {aicc_dict[best_model]:.2f}")

        return optimized_params_dict, aic_bic_dict, aicc_dict

# Usage
ts = np.array([0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120])
Vs = np.array([250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000])

analyzer = GrowthModelAnalyzer(ts, Vs)
analyzer.run_analysis()
