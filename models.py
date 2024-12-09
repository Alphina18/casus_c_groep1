import numpy as np
from scipy.integrate import solve_ivp

print("models.py is geladen")

class GrowthModel:  # super class
    def __init__(self, model, c=1.0, d=1.0):
        self.c = c
        self.d = d
        self.model = model

    def growth_rate(self, V):
        raise NotImplementedError("Deze methode moet worden geïmplementeerd door subklassen.")
    
    def euler_method(self, t):
        steps = 100
        V = 0                                     # Beginconditie
        dt = t / steps
        for _ in range(steps):
            dVdt = self.model.growth_rate(self, max(V, 1e-10)) # Differentiaalvergelijking
            V += dVdt * dt
        return V
    
    def euler_method_test(self, t, **params):
        steps = 1000
        V = params['V0']                                     # Beginconditie
        dt = t / steps
        for _ in range(steps):
            dVdt = self.model.growth_rate_test(self, V, params['d'] , params['c'] , params['V0'] , params['Vmax'] , params['Vmin'] ) # Differentiaalvergelijking
            V += dVdt * dt
        return V

    def mean_squared_error(self, data_ts, data_ys, **params):
        N = len(data_ts)
        total = 0.0
        for i in range(N):
            error = data_ys[i] - GrowthModel.euler_method_test(self, t=data_ts[i], **params)
            total += error * error
        return total / N

    def best_params(self, data_ts, data_ys, params):#
        print(params)
        deltas = {key: 1.0 for key in params}

        mse = GrowthModel.mean_squared_error(self, data_ts, data_ys, **params)
        while max(abs(delta) for delta in deltas.values()) > 1e-8:
            for key in params:
                new_params = params.copy()
                # Probeer de betreffende parameter the verhogen
                new_params[key] = params[key] + deltas[key]
                new_mse = GrowthModel.mean_squared_error(self, data_ts, data_ys, **new_params)
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= 1.2
                    continue
                # Probeer de betreffende parameter the verlagen
                new_params[key] = params[key] - deltas[key]
                new_mse = GrowthModel.mean_squared_error(self, data_ts, data_ys, **new_params)
                if new_mse < mse:
                    params = new_params
                    mse = new_mse
                    deltas[key] *= -1.0
                    continue
                # Verklein de stapgrootte
                deltas[key] *= 0.5
        return new_params

class LinearGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c

class LinearGrowth_test(GrowthModel):
    def growth_rate_test(self, V,d,c,V0, Vmax, Vmin):
        return c

class ExponentialGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * V

class MendelsohnGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * (V ** self.d)

class ExponentialDecayGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * (self.Vmax - V)

class LogisticGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * self.V * (self.V_max - self.V)

class MontrollGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * V * (self.Vmax ** self.d - V ** self.d)

class AlleeEffectGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * (V - self.Vmin) * (self.Vmax - V)

class LinearLimitedGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * (V / (V + self.d))

class SurfaceLimitedGrowth(GrowthModel):
    def __init__(self, c, d):
        super().__init__(c, d)

    def growth_rate(self, V):
        return self.c * (V / ((V + self.d) ** (1/3)))

class VonBertalanffyGrowth(GrowthModel):
    def __init__(self, c, d):
        super().__init__(c, d)

    def growth_rate(self, V):
        return self.c * (V ** (2/3)) - self.d * V

class GompertzGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * V * np.log(1.0 / V)
