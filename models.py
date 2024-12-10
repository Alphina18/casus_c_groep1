import numpy as np

"""
- meerdere solvers(heun, runge-kutta)
- meerdere keuzes van beste parameters krijgen(random optimalizatie)
- AIC, AICC en BIC
- eigen modellen toeveoegen

- grafiek met alle modellen en een df met de AIC, AICC en BIC waarden van alle modellen
"""

class GrowthModel:
    def __init__(self, solver=None, optimizer=None, evaluation=None):
        self.solver = solver
        self.optimizer = optimizer
        self.evaluation = evaluation

    required_params = set()  # Default empty set for base class

    def growth_rate(self, V):
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

    def euler_method_plot(self, t, params):
        steps = 100
        V = params["V0"]  # Initial condition
        dt = t / steps
        for _ in range(steps):
            if V <= 0:
                V = 1e-6  # Avoid zero or negative values
            dVdt = self.growth_rate(V, **params)
            V += dVdt * dt
        return V

    def mean_squared_error(self):
        N = len(self.tdata)
        total = 0.0
        for i in range(N):
            V_pred = self.euler_method(self.tdata[i]) # TODO: meerdere solver options
            error = self.vdata[i] - V_pred
            total += error ** 2
        return total / N

    def random_search(self):
        # TODO: implement random search
        pass

    def direct_search(self):
        # Only initialize the parameters relevant for this model
        params = {param: 1.0 for param in self.required_params}
        self.params = params
        deltas = {key: 1.0 for key in params}
        mse = self.mean_squared_error()

        while min(abs(delta) for delta in deltas.values()) > 1e-8:
            for key in params:
                new_params = params.copy()
                # Increase the parameter
                new_params[key] = params[key] + deltas[key]
                self.params = new_params
                new_mse = self.mean_squared_error()
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= 1.2
                    continue
                # Decrease the parameter
                new_params[key] = params[key] - deltas[key]
                self.params = new_params
                new_mse = self.mean_squared_error()
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= -1.0
                    continue
                # Reduce the step size
                deltas[key] *= 0.5
        return params
    
    def fit_data(self, data_ts, data_ys):
        self.vdata= data_ys
        self.tdata = data_ts
        if self.optimizer == "direct":
            params = self.direct_search()
        self.params = params

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

# Define required parameters for each model
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
        # Cap V to prevent overflow
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
        return self.params["c"] * (V / (V + self.params["d"]))


class SurfaceLimitedGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V / ((V + self.params["d"]) ** (1 / 3)))


class VonBertalanffyGrowth(GrowthModel):
    required_params = {"c", "d", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * (V ** (2 / 3)) - self.params["d"] * V


class GompertzGrowth(GrowthModel):
    required_params = {"c", "V0"}

    def growth_rate(self, V):
        return self.params["c"] * V * np.log(1.0 / V)
