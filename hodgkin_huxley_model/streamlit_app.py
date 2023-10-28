from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

from hodgkin_huxley_model.hodgkin_huxley import HodgkinHuxleyModel
from scipy.fft import fft, fftfreq


class StreamlitApp:
    def __init__(self, t_final: int = 100, dt: float = 0.2) -> None:
        """Initializes the Streamlit application for simulating the Hodgkin-Huxley model.

        Args:
            t_final (int, optional): The final time for the simulation. Defaults to 100.
            dt (float, optional): The time step size for the simulation. Defaults to 0.2.
        """
        self.t_final = t_final
        self.dt = dt
        self.V_init: float
        self.n_init: float = 0.3177
        self.m_init: float = 0.0530
        self.h_init: float = 0.5960

        self.model: HodgkinHuxleyModel

    def run(self):
        """Runs the Streamlit application for simulating the Hodgkin-Huxley model."""
        st.title("Hodgkin-Huxley Model Simulation")
        self._add_documentation()

        self._add_slider()

        self.model = HodgkinHuxleyModel(
            V_init=self.V_init,
            n_init=self.n_init,
            m_init=self.m_init,
            h_init=self.h_init,
            dt=self.dt,
            t_final=self.t_final,
        )

        I_inj = self._create_injection_current()

        self.model.simulate(I_inj)

        self._plot_voltage(I_inj)
        self._plot_currents()
        self._plot_fft()

    @staticmethod
    def _add_documentation():
        """Adds the documentation about the Hodgkin-Huxley model to the Streamlit app."""
        st.markdown(
            """
            The Hodgkin-Huxley model is a mathematical representation of the electrical 
            behavior of excitable cells, such as neurons. It describes the dynamics of 
            ion channels and membrane potential changes during the generation of action 
            potentials. In this simulation, we'll explore the behavior of a neuron's 
            membrane potential over time.

            The model is based on following equations:

            """
        )
        st.markdown("The membrane potential equation based on Coulumbs law:")
        st.latex(
            r"C_m\frac{dV}{dt} = I_{\text{inj}} - I_{\text{Na}} - I_{\text{K}} - I_{\text{L}}"
        )
        st.markdown("The sodium channel activation variable equation:")
        st.latex(r"")
        st.latex(r"\alpha_m(V) = 0.1\frac{V + 40}{1 - \exp(-0.1(V + 40))}")
        st.latex(r"\beta_m(V) = 4.0\exp(-0.0556(V + 65))")
        st.latex(r"\frac{dm}{dt} = \alpha_m(1 - m) - \beta_m m")

        st.markdown("The potassium channel activation variable equation:")
        st.latex(r"\alpha_n(V) = 0.01\frac{V + 55}{1 - \exp(-0.1(V + 55))}")
        st.latex(r"\beta_n(V) = 0.125\exp(-0.0125(V + 65))")
        st.latex(r"\frac{dn}{dt} = \alpha_n(1 - n) - \beta_n n")

        st.markdown("The sodium channel inactivation variable equation:")

        st.latex(r"\alpha_h(V) = 0.07\exp(-0.05(V + 65))")
        st.latex(r"\beta_h(V) = \frac{1}{1 + \exp(-0.1(V + 35))}")
        st.latex(r"\frac{dh}{dt} = \alpha_h(1 - h) - \beta_h h")

        st.markdown(
            r"Here, $V$ represents the membrane potential, $m$, $n$, and $h$ are the gating variables, and $I_{\text{inj}}$, $I_{\text{Na}}$, $I_{\text{K}}$, and $I_{\text{L}}$ denote the injected current, sodium current, potassium current, and leakage current, respectively. $C_m$ represents the membrane capacitance. The variables are functions of the membrane potential, and their dynamics are influenced by the voltage-dependent activation and inactivation properties of the ion channels."
        )

    def _add_slider(self):
        """Adds sliders for adjusting initial voltage and injection current to the Streamlit app."""
        self.V_init = st.slider(
            "Initial Voltage (mV)",
            -100.,
            100.,
            -64.9964,
            help="Initial Voltage (V_init): The starting membrane potential in mV.",
        )

        self.I_inj = st.slider(
            "Injection current [mA/(cm^2)]",
            0.0,
            0.15,
            0.3,
            help="Injection current (I_inj): The external current injected into the cell.",
        )

        self.inj_time = st.slider(
            "Time for injection current",
            0,
            self.t_final,
            (30, 33),
            help="Time for injection current (inj_time): The time interval during which the injection current is applied.",
        )

    def _create_injection_current(self) -> np.ndarray:
        """Creates the injection current based on the specified time interval.

        Returns:
            np.ndarray: The array representing the injection current.
        """
        # calculate injection current interval
        I_inj = np.ones_like(self.model.t) * self.I_inj
        start, stop = self.inj_time
        outside_interval = np.where(
            np.logical_or(self.model.t < start, self.model.t > stop)
        )
        I_inj[outside_interval] = 0

        return I_inj

    def _plot_voltage(self, I_inj):
        """Plots the voltage and the injected current over time.

        Args:
            I_inj: The injected current array.
        """
        # voltage figure
        v_figure = plt.figure()
        v_ax = v_figure.add_subplot()
        v_ax.plot(self.model.t, self.model.V, label="Voltage [mV]")
        v_ax.plot(self.model.t, I_inj * 100, label=f"Injection current {max(I_inj)} [µA/cm^2]")
        # v_ax.plot(t, I_inj, label="Injection current")
        v_ax.set_ylabel("Voltage [mV]")
        v_ax.set_xlabel("time [mS]")
        v_ax.set_title("Hodgkin-Huxley Model Simulation")
        # v_ax.set_ylim(-60, 60)
        v_ax.grid()
        v_ax.legend()

        st.pyplot(v_figure)

    def _plot_currents(self):
        """Plots the individual currents n, m, and h over time."""
        if not st.button("individual currents"):
            return

        fig, (n_ax, m_ax, h_ax) = plt.subplots(
            3, 1, figsize=(10, 12), sharex=True
        )
        

        n_ax.plot(self.model.t, self.model.n, label="n: sodium current [mA]")
        n_ax.set_ylabel("n: sodium current")
        n_ax.grid()

        m_ax.plot(self.model.t, self.model.m, label="m: potassium current [mA]")
        m_ax.set_ylabel("m: potassium current")
        m_ax.grid()

        h_ax.plot(self.model.t, self.model.h, label="h: leak current [mA]")
        h_ax.set_xlabel("Time (ms)")
        h_ax.set_ylabel("h: leak_current")
        h_ax.grid()

        h_ax.set_xlabel("time [mS]")

        st.pyplot(fig)

    def _plot_fft(self):
        """Performs the Fourier analysis and plots the Fourier transform of the membrane potential."""
        if not st.button("Fourier Analysis"):
            return

        yf = fft(self.model.V)
        N = len(self.model.t)
        xf = fftfreq(N, self.dt)[: N // 2]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
        ax.grid()
        
        st.pyplot(fig)