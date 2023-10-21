from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

from hodgkin_huxley_model.hodgkin_huxley import HodgkinHuxleyModel


class StreamlitApp:
    def run(self):
        st.title("Hodgkin-Huxley Model Simulation")


        V_init = st.slider("Initial Voltage (mV)", -100, 100, -65)
        n_init = 0.317
        m_init = 0.05
        h_init = 0.6
        t_final = 50

        I_inj = st.slider("Injection current [µA/(cm^2)]", 0.0, 10., 2.)
        inj_time = st.slider("Time for injection current", 0, t_final, (30, 33))
        
        model = HodgkinHuxleyModel(
            V_init=V_init, n_init=n_init, m_init=m_init, h_init=h_init, dt=0.1, t_final=t_final
        )

        I_inj = np.ones_like(model.t) * I_inj
        start, stop = inj_time
        outside_interval = np.where(np.logical_or(model.t<start, model.t>stop))
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