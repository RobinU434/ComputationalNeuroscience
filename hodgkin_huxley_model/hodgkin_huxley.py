import numpy as np

class HodgkinHuxleyModel:
    def __init__(self, V_init, n_init, m_init, h_init, dt, t_final):
        self.V_init = V_init
        self.n_init = n_init
        self.m_init = m_init
        self.h_init = h_init
        self.dt = dt
        self.t_final = t_final

        self.C_m = 1.0  # membrane capacitance, in uF/cm^2
        self.g_Na = 120.0  # maximum conductances, in mS/cm^2
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0  # Nernst reversal potentials, in mV
        self.E_K = -77.0
        self.E_L = -54.387

    def simulate(self):

        t = np.arange(0, self.t_final, self.dt)

        V = np.zeros(len(t))
        n = np.zeros(len(t))
        m = np.zeros(len(t))
        h = np.zeros(len(t))

        V[0] = self.V_init
        n[0] = self.n_init
        m[0] = self.m_init
        h[0] = self.h_init

        for i in range(1, len(t)):
            alpha_n = 0.01 * (V[i - 1] + 55) / (1 - np.exp(-0.1 * (V[i - 1] + 55)))
            beta_n = 0.125 * np.exp(-0.0125 * (V[i - 1] + 65))
            n[i] = n[i - 1] + self.dt * (alpha_n * (1 - n[i - 1]) - beta_n * n[i - 1])

            alpha_m = 0.1 * (V[i - 1] + 40) / (1 - np.exp(-0.1 * (V[i - 1] + 40)))
            beta_m = 4.0 * np.exp(-0.0556 * (V[i - 1] + 65))
            m[i] = m[i - 1] + self.dt * (alpha_m * (1 - m[i - 1]) - beta_m * m[i - 1])

            alpha_h = 0.07 * np.exp(-0.05 * (V[i - 1] + 65))
            beta_h = 1 / (1 + np.exp(-0.1 * (V[i - 1] + 35)))
            h[i] = h[i - 1] + self.dt * (alpha_h * (1 - h[i - 1]) - beta_h * h[i - 1])

            I_Na = self.g_Na * (m[i - 1] ** 3) * h[i - 1] * (V[i - 1] - self.E_Na)
            I_K = self.g_K * (n[i - 1] ** 4) * (V[i - 1] - self.E_K)
            I_L = self.g_L * (V[i - 1] - self.E_L)
            I_inj = 10  # injected current
            V[i] = V[i - 1] + self.dt * ((I_inj - I_Na - I_K - I_L) / self.C_m)

        return t, V, n, m, h
