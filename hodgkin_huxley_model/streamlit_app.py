from matplotlib import pyplot as plt
import streamlit as st

from hodgkin_huxley_model.hodgkin_huxley import HodgkinHuxleyModel


class StreamlitApp:
    def run(self):
        st.title("Hodgkin-Huxley Model Simulation")

        model = HodgkinHuxleyModel(
            V_init=-70, n_init=0.317, m_init=0.05, h_init=0.6, dt=0.01, t_final=50
        )

        V_init = st.slider("Initial Voltage (mV)", -100, 100, model.V_init)
        n_init = st.slider("Initial n", 0.0, 1.0, model.n_init)
        m_init = st.slider("Initial m", 0.0, 1.0, model.m_init)
        h_init = st.slider("Initial h", 0.0, 1.0, model.h_init)

        # if st.button("Run Simulation"):
        model.V_init = V_init
        model.n_init = n_init
        model.m_init = m_init
        model.h_init = h_init

        t, V, n, m, h = model.simulate()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

        ax1.plot(t, V, label="Voltage (mV)")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_title("Hodgkin-Huxley Model Simulation")
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
