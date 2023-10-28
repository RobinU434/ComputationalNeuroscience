from typing import Tuple
import numpy as np


class HodgkinHuxleyModel:
    def __init__(self, V_init, n_init, m_init, h_init, dt, t_final):
        """Initializes the Hodgkin-Huxley model with the specified parameters.

        Args:
            V_init (float): Initial membrane potential.
            n_init (float): Initial value for the gating variable n.
            m_init (float): Initial value for the gating variable m.
            h_init (float): Initial value for the gating variable h.
            dt (float): Time step size for the simulation.
            t_final (float): Final time for the simulation.
        """
        self.V_init = V_init
        self.n_init = n_init
        self.m_init = m_init
        self.h_init = h_init
        self.dt = dt
        self.t_final = t_final

        # membrane capacitance, in mF/cm^2
        self.C_m = 0.01
        # maximum conductances, in mS/cm^2
        self.g_Na = 1.200  # Sodium
        self.g_K = 0.360  # Potassium
        self.g_L = 0.003  # leak
        # Nernst reversal potentials, in mV
        self.E_Na = 50.0
        # self.E_Na = -115
        self.E_K = -77.0
        # self.E_K = 12.0
        self.E_L = -54.387
        # self.E_L = -10.

        self.t = np.arange(0, self.t_final, self.dt)

        self.V = np.empty(len(self.t))
        self.n = np.empty(len(self.t))
        self.m = np.empty(len(self.t))
        self.h = np.empty(len(self.t))

    def _n(self, V_prev: float, n_prev: float) -> float:
        """Calculates the gating variable n which represents the probability that the voltage-gated potassium channels are open.

        Args:
            V_prev (float): Previous membrane potential.
            n_prev (float): Previous value for the gating variable n.

        Returns:
            float: The updated value for the gating variable n.
        """
        # alpha_n determines the rate of transfer from outside to inside
        alpha_n = 0.01 * (V_prev + 55) / (1 - np.exp(-(V_prev + 55) / 10))
        # alpha_n =  (0.01 * (10 - V_prev)) / (np.exp((10 - V_prev) / 10) - 1)
        # alpha_n determines the rate of transfer from inside to outside,
        beta_n = 0.125 * np.exp(-(V_prev + 65) / 80)
        # beta_n = 0.125 * np.exp(- V_prev / 80)
        n = self.g(alpha=alpha_n, beta=beta_n, prev_value=n_prev)
        return n

    def _m(self, V_prev: float, m_prev: float) -> float:
        """Calculates the gating variable m which represents the probability that the voltage-gated sodium channels are open.

        Args:
            V_prev (float): Previous membrane potential.
            m_prev (float): Previous value for the gating variable m.

        Returns:
            float: The updated value for the gating variable m.
        """
        alpha_m = 0.1 * (V_prev + 40) / (1 - np.exp(-(V_prev + 40) / 10))
        alpha_m = 0.1 * (V_prev + 40) / (1 - np.exp(-(V_prev + 40) / 10))
        beta_m = 4.0 * np.exp(-0.0556 * (V_prev + 65))
        beta_m = 4.0 * np.exp(-0.0556 * (V_prev + 65))
        m = self.g(alpha=alpha_m, beta=beta_m, prev_value=m_prev)

        return m

    def _h(self, V_prev: float, h_prev: float) -> float:
        """Calculates the gating variable h.

        Args:
            V_prev (float): Previous membrane potential.
            h_prev (float): Previous value for the gating variable h.

        Returns:
            float: The updated value for the gating variable h.
        """
        alpha_h = 0.07 * np.exp(-0.05 * (V_prev + 65))
        alpha_h = 0.07 * np.exp(-0.05 * (V_prev + 65))
        beta_h = 1 / (1 + np.exp(-0.1 * (V_prev + 35)))
        beta_h = 1 / (1 + np.exp(-0.1 * (V_prev + 35)))
        h = self.g(alpha=alpha_h, beta=beta_h, prev_value=h_prev)

        return h

    def _v(
        self, m_prev: float, h_prev: float, n_prev: float, v_prev: float, I_inj: float
    ) -> float:
        """Calculates the membrane potential, the main variable that governs the behavior of the system.

        Args:
            m_prev (float): Previous value for the gating variable m.
            h_prev (float): Previous value for the gating variable h.
            n_prev (float): Previous value for the gating variable n.
            v_prev (float): Previous membrane potential.
            I_inj (float): Injected current.

        Returns:
            float: The updated membrane potential.
        """
        g_Na = self.g_Na * (np.power(m_prev, 3)) * h_prev
        g_K = self.g_K * (np.power(n_prev, 4))
        g_sum = g_Na + g_K + self.g_L
        
        I_Na = g_Na * self.E_Na
        I_K = g_K * self.E_K
        I_L = self.g_L * self.E_L

        tau = self.C_m / g_sum
        v_inf = (I_Na + I_K + I_L + I_inj) / g_sum
        v = v_inf + (v_prev - v_inf) * np.exp(-self.dt / tau)
        return v

    def simulate(self, I_inj: np.ndarray):
        """Simulates the Hodgkin-Huxley model based on the injected current array.

        Args:
            I_inj (np.ndarray): Array of injected currents.
        """
        self.V[0] = self.V_init
        self.n[0] = self.n_init
        self.m[0] = self.m_init
        self.h[0] = self.h_init
        for i in range(1, len(self.t)):
            self.V[i] = self._v(
                m_prev=self.m[i - 1],
                h_prev=self.h[i - 1],
                n_prev=self.n[i - 1],
                v_prev=self.V[i - 1],
                I_inj=I_inj[i - 1],
            )

            self.n[i] = self._n(V_prev=self.V[i], n_prev=self.n[i - 1])
            self.m[i] = self._m(V_prev=self.V[i], m_prev=self.m[i - 1])
            self.h[i] = self._h(V_prev=self.V[i], h_prev=self.h[i - 1])
        
    def g(self, alpha: float, beta: float, prev_value: float) -> float:
        """Calculates the conductance based on the provided alpha, beta, and previous value.

        Args:
            alpha (float): Alpha value.
            beta (float): Beta value.
            prev_value (float): Previous conductance value.

        Returns:
            float: The updated conductance value.
        """
        tau = 1 / (alpha + beta)
        g_inf = alpha * tau
        result = g_inf + (prev_value - g_inf) * np.exp(-self.dt / tau)
        return result
