#%%
from scipy.interpolate import interp1d
import numpy as np

def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

class CreateCase:
    def __init__(self, hs, tz, m, c, k, k1, t, G, altered=False):
        self.hs = hs
        self.tz = tz
        self.m = m
        self.k = k
        self.k1 = k1
        self.c = c
        self.G = G
        self.w = np.arange(0.01, 2, .001)
        self.altered = altered
        self.s_eta = self.get_s_eta()
        self.t = t
        self.yt = self.get_yt(t)
        self.interpolador = interp1d(t, 
                                     self.yt, 
                                     bounds_error=False,
                                     fill_value="extrapolate")

    def get_s_eta(self):
        """ Calculate wave spectrum """
        p1 = 4 * np.pi**3 * self.hs**2 / self.w**5 / self.tz**4
        pote = - 16 * np.pi**3 / self.w**4 / self.tz**4
        return p1 * np.exp(pote)

    def get_yt(self, t):
        """ calculo de y(t) """
        s = self.s_eta
        dw = np.diff(self.w).mean()
        n = len(s)
        # Initialize vector
        vec = np.array([t] * n).T
        # randon phase
        ang = vec * self.w
        phi = np.random.uniform(high=2*np.pi, size=ang.shape)
        c = np.sqrt(2 * s * dw)
        if self.altered:
            result = c * np.cos(ang + phi) + self.G * self.hs**2
        else:
            result = c * np.cos(ang + phi) + self.G * c**2
        return np.sum(result, axis=1)
    
    def get_xt(self):
        """ Calculate derivative for ode int """
        def dx_dt(x, t):
            
            return np.array([x[1], 
                             - self.c*x[1] / self.m \
                             - self.k*x[0] / self.m \
                             - self.k1*x[0]**3 / self.m \
                             + self.interpolador(t) / self.m])

        x0 = [0, 0]
        return rungekutta4(dx_dt, x0, self.t)


if __name__ == "__main__":
    # Testing
    t = np.arange(0, 3 * 10, .125)
    hs = 7.8
    tz = 11
    m = .05
    k = 2
    k1 = 0.3
    c = .2
    
    case = CreateCase(hs, tz, m, c, k, k1, t, 1)
    
#%%