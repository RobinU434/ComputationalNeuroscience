from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

from hodgkin_huxley_model.hodgkin_huxley import HodgkinHuxleyModel


class StreamlitApp:
    def run(self):
        st.title("Hodgkin-Huxley Model Simulation")

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
        
        V_init = st.slider(
            "Initial Voltage (mV)",
            -100,
            100,
            -65,
            help="Initial Voltage (V_init): The starting membrane potential in mV.",
        )
        n_init = 0.317
        m_init = 0.05
        h_init = 0.6
        t_final = 50

        I_inj = st.slider(
            "Injection current [µA/(cm^2)]",
            0.0,
            10.0,
            2.0,
            help="Injection current (I_inj): The external current injected into the cell.",
        )
        inj_time = st.slider(
            "Time for injection current",
            0,
            t_final,
            (30, 33),
            help="Time for injection current (inj_time): The time interval during which the injection current is applied.",
        )

        

        model = HodgkinHuxleyModel(
            V_init=V_init,
            n_init=n_init,
            m_init=m_init,
            h_init=h_init,
            dt=0.1,
            t_final=t_final,
        )

        I_inj = np.ones_like(model.t) * I_inj
        start, stop = inj_time
        outside_interval = np.where(np.logical_or(model.t < start, model.t > stop))
        I_inj[outside_interval] = 0
        t, V, n, m, h = model.simulate(I_inj)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

        ax1.plot(t, V, label="Voltage (mV)")
        # ax1.plot(t, I_inj, label="Injection current")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_title("Hodgkin-Huxley Model Simulation")
        # ax1.set_ylim(-60, 60)
        ax1.grid()

        ax2.plot(t, n, label="n")
        ax2.set_ylabel("n")
        ax2.grid()

        ax3.plot(t, m, label="m")
        ax3.set_ylabel("m")
        ax3.grid()

        ax4.plot(t, h, label="h")
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("h")
        ax4.grid()

        st.pyplot(fig)
