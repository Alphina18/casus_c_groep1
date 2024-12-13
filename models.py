import numpy as np
from skopt import gp_minimize
import random
"""
- meerdere solvers(heun, runge-kutta)
- meerdere keuzes van beste parameters krijgen(random optimalizatie)
- AIC, AICC en BIC
- eigen modellen toeveoegen

- grafiek met alle modellen en een df met de AIC, AICC en BIC waarden van alle modellen
"""

class GrowthModel:
    required_params = set()  # Default empty set for base class

    def __init__(self, solver=None, optimizer=None, evaluation=None):
        if solver == "rungekutta":
            self.solver = self.runge_kutta
        elif solver == "heun":
            self.solver = self.heun_method
        elif solver == "euler":
            self.solver = self.euler_method
        else:
            raise NotImplementedError("The variable solver is None.")
        
        if optimizer == "direct":
            self.optimizer = self.direct_search
        elif optimizer == "bayesian":
            self.optimizer = self.bayesian_search
        elif optimizer == "random":
            self.optimizer = self.random_search
        else:
            raise NotImplementedError("The variable optimizer is None.")
        
        if evaluation == "AIC":
            self.evaluation = self.eval_aic
        elif evaluation == "AICC":
            self.evaluation = self.eval_aicc
        elif evaluation == "BIC":
            self.evaluation = self.eval_bic
        else:
            raise NotImplementedError("The variable evaluation is None.")

    def growth_rate(self, V):
        """To be implemented by subclasses."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def euler_method(self, t):
        steps = int(abs(t) / 0.01) + 1 
        V = self.params["V0"]  # Initial condition
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            if V >= 1e10: # Avoid big numbers
                V = 1e10
            dVdt = self.growth_rate(V)
            V += dVdt * dt
        return V
    
    def heun_method(self, t):
        steps = int(abs(t) / 0.01) + 1 
        V = self.params["V0"]                            # Beginconditie
        dt = t / steps
        for _ in range(steps):
            # Tijdelijke stappen:
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            if V >= 1e10: # Avoid big numbers
                V = 1e10
            dVdt1 = self.growth_rate(V)                 # Differentiaalvergelijking stap 1
            V1 = V + dVdt1 * dt
            dVdt2 = self.growth_rate(V1)                # Differentiaalvergelijking stap 2
            # Definitieve stap:
            dydt = (dVdt1 + dVdt2) / 2.0
            V += dydt * dt
        return V
    
    def runge_kutta(self, t):
        steps = int(abs(t) / 0.01) + 1 
        V = self.params["V0"]                            # Beginconditie
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            if V >= 1e10: # Avoid big numbers
                V = 1e10
            dVdt1 = self.growth_rate(V)                # Differentiaalvergelijking stap 1
            V1 = V + dVdt1 * 0.5 * dt
            dVdt2 = self.growth_rate(V1)                # Differentiaalvergelijking stap 2
            V2 = V + dVdt2 * 0.5 * dt
            dVdt3 = self.growth_rate(V2)                 # Differentiaalvergelijking stap 3
            V3 = V + dVdt3 * dt
            dydt4 = self.growth_rate(V3)                # Differentiaalvergelijking stap 4
            # Definitieve stap:
            dydt = (dVdt1 + 2.0 * dVdt2 + 2.0 * dVdt3 + dydt4) / 6.0
            V += dydt * dt
        return V

    def mean_squared_error(self):
        N = len(self.tdata)
        total = 0.0
        for i in range(N):
            V_pred = self.solver(self.tdata[i])  # TODO: meerdere solver opties
            error = self.vdata[i] - V_pred
            #print(f"V_pred: {V_pred}, vdata: {self.vdata[i]}, error: {error}")  # Debugging
            total += error ** 2
        return total / N

    def random_search(self):
        # Initialize parameters (this could be changed depending on the model)
        params = {key: 0.0 for key in self.required_params}  # Initial guess for parameters
        mse = self.mean_squared_error()
        
        tries = 0
        while tries < 1000:
            tries += 1
            new_params = {key: val + (random() - 0.5) for key, val in params.items()}
            self.params = new_params
            new_mse = self.mean_squared_error()
            
            # If the new MSE is better, keep the parameters
            if new_mse < mse:
                params = new_params
                mse = new_mse
                tries = 0  # Reset tries if we find a better set of parameters
        self.params = params

    def bayesian_search(self):
        # Define the search space for the parameters
        search_space = [(0.01, 10.0) for _ in self.required_params]  # Example bounds for parameters
        def objective_function(params):
            # Map parameters to the model's params dictionary
            for i, param in enumerate(self.required_params):
                self.params[param] = params[i]
            return self.evaluation()

        # Perform Bayesian optimization
        res = gp_minimize(objective_function, search_space, n_calls=15)
        # Update the model parameters with the best found values
        for i, param in enumerate(self.required_params):
            self.params[param] = res.x[i]

    def grid_search(self):
        # TODO: implement grid search
        pass

    def direct_search(self):
        # Only initialize the parameters relevant for this model
        params = {param: 1.0 for param in self.required_params}
        self.params = params
        deltas = {key: 1.0 for key in params}
        mse = self.evaluation()

        while min(abs(delta) for delta in deltas.values()) > 1e-8:
            for key in params:
                new_params = params.copy()
                # Increase the parameter
                new_params[key] = params[key] + deltas[key]
                self.params = new_params
                new_mse = self.evaluation()
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= 1.2
                    continue
                # Decrease the parameter
                new_params[key] = params[key] - deltas[key]
                self.params = new_params
                new_mse = self.evaluation()
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= -1.0
                    continue
                # Reduce the step size
                deltas[key] *= 0.5
        self.params = params
    
    def fit_data(self, data_ts, data_ys):
        self.vdata= data_ys
        self.tdata = data_ts
        self.params = {param: 1.0 for param in self.required_params}
        self.optimizer()

    def eval_aic(self):
        k = len(self.params)
        n = len(self.vdata)
        return n * np.log(self.mean_squared_error()) + 2*k

    def eval_aicc(self):
        k = len(self.params)
        n = len(self.vdata)
        return self.eval_aic() + (2*k + (k+1)) / (n-k-1)

    def eval_bic(self):
        k = len(self.params)
        n = len(self.vdata)
        return n * np.log(self.mean_squared_error()) + np.log(n)*k
    

class LinearGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"]
    
    def __str__(self):
        return "c"

    def __repr__(self):
        return "LinearGrowth"

class ExponentialGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V

    def __str__(self):
        return "c * V"

    def __repr__(self):
        return "ExponentialGrowth"

class MendelsohnGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        # Cap V to prevent overflow
        return self.params["c"] * (V ** self.params["d"])

    def __str__(self):
        return "c * V^d"

    def __repr__(self):
        return "MendelsohnGrowth"

class ExponentialDecayGrowth(GrowthModel):
    required_params = {"c", "Vmax", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (self.params["Vmax"] - V)

    def __str__(self):
        return "c * (Vmax - V)"

    def __repr__(self):
        return "ExponentialDecayGrowth"


class LogisticGrowth(GrowthModel):
    required_params = {"c", "V0", "Vmax"}

    def growth_rate(self, V):
        return self.params["c"] * V * (self.params["Vmax"] - V)

    def __str__(self):
        return "c * V * (Vmax - V)"

    def __repr__(self):
        return "LogisticGrowth"


class MontrollGrowth(GrowthModel):
    required_params = {"c", "d", "Vmax", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V * (self.params["Vmax"] ** self.params["d"] - V ** self.params["d"])

    def __str__(self):
        return "c * V * (Vmax^d - V^d)"

    def __repr__(self):
        return "MontrollGrowth"


class AlleeEffectGrowth(GrowthModel):
    required_params = {"c", "d", "Vmax", "Vmin", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V - self.params["Vmin"]) * (self.params["Vmax"] - V)

    def __str__(self):
        return "c * (V - Vmin) * (Vmax - V)"

    def __repr__(self):
        return "AlleeEffectGrowth"


class LinearLimitedGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V / (V + self.params["d"]))

    def __str__(self):
        return "c * (V / (V + d))"

    def __repr__(self):
        return "LinearLimitedGrowth"


class SurfaceLimitedGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V / ((V + self.params["d"]) ** (1 / 3)))

    def __str__(self):
        return "c * (V / ((V + d)^(1/3)))"

    def __repr__(self):
        return "SurfaceLimitedGrowth"


class VonBertalanffyGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V ** (2 / 3)) - self.params["d"] * V

    def __str__(self):
        return "c * (V^(2/3)) - d * V"

    def __repr__(self):
        return "VonBertalanffyGrowth"


class GompertzGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V * np.log(1.0 / V)

    def __str__(self):
        return "c * V * ln(1/V)"

    def __repr__(self):
        return "GompertzGrowth"

class CombinedModel(GrowthModel):
    required_params = {"c", "V0", "d", "Vmax"}

    def growth_rate(self, V):
        return self.params["c"] * (V ** (2 / 3)) * (1 - (V / self.params["Vmax"])) - self.params["d"] * V
    
    def __str__(self):
        return "c * (V^(2/3)) * (1 - V/V_max) - d * V"

    def __repr__(self):
        return "CombinedModel"