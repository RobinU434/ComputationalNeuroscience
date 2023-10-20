import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley model parameters
C_m  = 1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conductances, in mS/cm^2
g_K  = 36.0
g_L  = 0.3
E_Na = 50.0  # Nernst reversal potentials, in mV
E_K  = -77.0
E_L  = -54.387

@st.cache
def run_hodgkin_huxley(V_init, n_init, m_init, h_init, dt, t_final):
    t = np.arange(0, t_final, dt)

    V = np.zeros(len(t))
    n = np.zeros(len(t))
    m = np.zeros(len(t))
    h = np.zeros(len(t))

    V[0] = V_init
    n[0] = n_init
    m[0] = m_init
    h[0] = h_init

    for i in range(1, len(t)):
        alpha_n = 0.01 * (V[i - 1] + 55) / (1 - np.exp(-0.1 * (V[i - 1] + 55)))
        beta_n = 0.125 * np.exp(-0.0125 * (V[i - 1] + 65))
        n[i] = n[i - 1] + dt * (alpha_n * (1 - n[i - 1]) - beta_n * n[i - 1])

        alpha_m = 0.1 * (V[i - 1] + 40) / (1 - np.exp(-0.1 * (V[i - 1] + 40)))
        beta_m = 4.0 * np.exp(-0.0556 * (V[i - 1] + 65))
        m[i] = m[i - 1] + dt * (alpha_m * (1 - m[i - 1]) - beta_m * m[i - 1])

        alpha_h = 0.07 * np.exp(-0.05 * (V[i - 1] + 65))
        beta_h = 1 / (1 + np.exp(-0.1 * (V[i - 1] + 35)))
        h[i] = h[i - 1] + dt * (alpha_h * (1 - h[i - 1]) - beta_h * h[i - 1])

        I_Na = g_Na * (m[i - 1] ** 3) * h[i - 1] * (V[i - 1] - E_Na)
        I_K = g_K * (n[i - 1] ** 4) * (V[i - 1] - E_K)
        I_L = g_L * (V[i - 1] - E_L)
        I_inj = 10  # injected current
        V[i] = V[i - 1] + dt * ((I_inj - I_Na - I_K - I_L) / C_m)

    return t, V, n, m, h

# Streamlit UI
st.title('Hodgkin-Huxley Model Simulation')

V_init = st.slider('Initial Voltage (mV)', -100, 100, -70)
n_init = st.slider('Initial n', 0.0, 1.0, 0.317)
m_init = st.slider('Initial m', 0.0, 1.0, 0.05)
h_init = st.slider('Initial h', 0.0, 1.0, 0.6)
dt = 0.01
t_final = 50

if st.button('Run Simulation'):
    t, V, n, m, h = run_hodgkin_huxley(V_init, n_init, m_init, h_init, dt, t_final)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    ax1.plot(t, V, label='Voltage (mV)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Hodgkin-Huxley Model Simulation')
    ax1.legend()

    ax2.plot(t, n, label='n')
    ax2.set_ylabel('n')
    ax2.legend()

    ax3.plot(t, m, label='m')
    ax3.set_ylabel('m')
    ax3.legend()

    ax4.plot(t, h, label='h')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('h')
    ax4.legend()

    st.pyplot(fig)

