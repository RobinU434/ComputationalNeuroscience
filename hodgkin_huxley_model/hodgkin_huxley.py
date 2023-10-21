import numpy as np

class HodgkinHuxleyModel:
    def __init__(self, V_init, n_init, m_init, h_init, dt, t_final):
        self.V_init = V_init
        self.n_init = n_init
        self.m_init = m_init
        self.h_init = h_init
        self.dt = dt
        self.t_final = t_final
        
        # membrane capacitance, in µF/cm^2
        self.C_m = 1.0
        # maximum conductances, in mS/cm^2
        self.g_Na = 120.0  # Sodium
        self.g_K = 36.0  # Potassium
        self.g_L = 0.3  # leak
        # Nernst reversal potentials, in mV
        self.E_Na = 50.0  
        self.E_K = -77.0  
        self.E_L = -54.387

        self.t = np.arange(0, self.t_final, self.dt)

        self.V = np.empty(len(self.t))
        self.n = np.empty(len(self.t))
        self.m = np.empty(len(self.t))
        self.h = np.empty(len(self.t))

        self.V[0] = self.V_init
        self.n[0] = self.n_init
        self.m[0] = self.m_init
        self.h[0] = self.h_init

    def _n(self, V_prev: float, n_prev: float) -> float:
        """Represents the probability that the voltage-gated potassium channels are open. These channels are responsible for the outward movement of potassium ions, contributing to the repolarization of the cell membrane.

        Args:
            V_prev (float): _description_
            n_prev (float): _description_

        Returns:
            float: 
        """
    
        alpha_n = 0.01 * (V_prev + 55) / (1 - np.exp(-0.1 * (V_prev + 55)))
        beta_n = 0.125 * np.exp(-0.0125 * (V_prev + 65))
        n = n_prev + self.dt * (alpha_n * (1 - n_prev) - beta_n * n_prev)
        return n
    

    def _m(self, V_prev: float, m_prev: float) -> float:
        """Represents the probability that the voltage-gated sodium channels are open. These channels are responsible for the inward movement of sodium ions, which leads to the depolarization of the cell membrane.

        Args:
            V_prev (float): _description_
            m_prev (float): _description_

        Returns:
            float: _description_
        """
        alpha_m = 0.1 * (V_prev + 40) / (1 - np.exp(-0.1 * (V_prev + 40)))
        beta_m = 4.0 * np.exp(-0.0556 * (V_prev + 65))
        m = m_prev + self.dt * (alpha_m * (1 - m_prev) - beta_m * m_prev)
        return m

    def _h(self, V_prev: float, h_prev: float) -> float:
        """_summary_

        Args:
            V_prev (float): _description_
            h_prev (float): _description_

        Returns:
            float: _description_
        """
        alpha_h = 0.07 * np.exp(-0.05 * (V_prev + 65))
        beta_h = 1 / (1 + np.exp(-0.1 * (V_prev + 35)))
        h = h_prev + self.dt * (alpha_h * (1 - h_prev) - beta_h * h_prev)
        return h
    
    def _v(self, m_prev: float, h_prev: float, n_prev: float, v_prev: float) -> float:
        """ represents the membrane potential, or the voltage across the cell membrane. It is the main variable that governs the behavior of the system.

        Args:
            m_prev (float): _description_
            h_prev (float): _description_
            n_prev (float): _description_
            v_prev (float): _description_

        Returns:
            _type_: _description_
        """
        I_Na = self.g_Na * (m_prev ** 3) * h_prev * (v_prev - self.E_Na)
        I_K = self.g_K * (n_prev ** 4) * (v_prev - self.E_K)
        I_L = self.g_L * (v_prev - self.E_L)
        I_inj = 10  # injected current
        v = v_prev + self.dt * ((I_inj - I_Na - I_K - I_L) / self.C_m)
        return v

    def simulate(self):
        for i in range(1, len(self.t)):
            self.n[i] = self._n(V_prev=self.V[i - 1], n_prev=self.n[i - 1])
            self.m[i] = self._m(V_prev=self.V[i - 1], m_prev=self.m[i - 1])
            self.h[i] = self._h(V_prev=self.V[i - 1], h_prev=self.h[i - 1])
            self.V[i] = self._v(m_prev=self.m[i - 1], h_prev=self.h[i - 1], n_prev=self.n[i - 1], v_prev=self.V[i - 1])
        return self.t, self.V, self.n, self.m, self.h
