import numpy as np
from scipy.integrate import solve_ivp

print("models.py is geladen")

class GrowthModel:  # super class
    def __init__(self, c, t=None, Vmax=None, Vmin=None, d=None, f=None, V0=None):
        self.c = c  
        self.Vmax = Vmax 
        self.Vmin = Vmin  
        self.d = d  
        self.f = f
        self.V0 = V0
        self.t = t

    def growth_rate(self, V):
        raise NotImplementedError("Deze methode moet worden geïmplementeerd door subklassen.")
    
    def euler_method(self):
        steps = 10
        y = 0.0                             # Beginconditie
        dt = self.t / steps
        for step in range(steps):
            dydt = y + 1.0                  # Differentiaalvergelijking
            y += dydt * dt
        return y

    def rungekutta_method(self):
        y = self.V0                              # Beginconditie
        steps = int(abs(self.t) / 0.01) + 1      # Hoeveel tijdstappen van ~0.01 zijn nodig
        dt = self.t / steps
        for step in range(steps):
            # Tijdelijke stappen:
            dydt1 = a * y + b               # Differentiaalvergelijking stap 1
            y1 = y + dydt1 * 0.5 * dt
            dydt2 = a * y1 + b              # Differentiaalvergelijking stap 2
            y2 = y + dydt2 * 0.5 * dt
            dydt3 = a * y2 + b              # Differentiaalvergelijking stap 3
            y3 = y + dydt3 * dt
            dydt4 = a * y3 + b              # Differentiaalvergelijking stap 4
            # Definitieve stap:
            dydt = (dydt1 + 2.0 * dydt2 + 2.0 * dydt3 + dydt4) / 6.0
            y += dydt * dt
        return y

    def other_method(self):
        pass


class LinearGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c

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
        return self.c * V * (self.Vmax - V)

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
    def growth_rate(self, V):
        return self.c * (V / ((V + self.d) ** (1/3)))

class VonBertalanffyGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * (V ** (2/3)) - self.d * V

class GompertzGrowth(GrowthModel):
    def growth_rate(self, V):
        return self.c * V * np.log(self.Vmax / V)
