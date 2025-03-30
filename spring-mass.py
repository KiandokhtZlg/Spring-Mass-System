from IPython.display import Image
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Display image 
Image('photo.png')

# Constants
k1 = k2 = k3 = k4 = 1 
k5 = 0.3
m1 = m2 = 1
l0 = 1

# Define system of differential equations
def eqs(t, P):
    """
    Defines the equations of motion for a system of two masses and springs.
    P = [u1, x1, u2, x2, w1, y1, w2, y2]
    """
    dP = np.zeros_like(P)
    
    x1, y1, x2, y2 = P[1], P[5], P[3], P[7]
    u1, w1, u2, w2 = P[0], P[4], P[2], P[6]
    
    # Compute forces using temporary variables for readability
    force_x1 = - (k2 * x1 / m1) * (1 - l0 / (x1**2 + (y1 + l0)**2)**0.5)
    force_x1 += - (k1 * x1 / m1) * (1 - l0 / (x1**2 + (l0 - y1)**2)**0.5)
    force_x1 += (k5 * (l0 - x1 + x2) / m1) * (1 - l0 / ((l0 - x1 + x2)**2 + (y2 - y1)**2)**0.5)
    
    force_x2 = - (k4 * x2 / m2) * (1 - l0 / (x2**2 + (y2 + l0)**2)**0.5)
    force_x2 += - (k3 * x2 / m2) * (1 - l0 / (x2**2 + (l0 - y2)**2)**0.5)
    force_x2 += - (k5 * (l0 - x1 + x2) / m2) * (1 - l0 / ((l0 - x1 + x2)**2 + (y2 - y1)**2)**0.5)
    
    dP[0] = force_x1  # du1/dt
    dP[1] = u1  # dx1/dt
    dP[2] = force_x2  # du2/dt
    dP[3] = u2  # dx2/dt
    dP[4] = w1  # dw1/dt
    dP[5] = w1  # dy1/dt
    dP[6] = w2  # dw2/dt
    dP[7] = w2  # dy2/dt
    
    return dP

# Initial conditions
P_0 = np.array([0.6, 0, 0, 0, 0.4, 0.2, 0, 0])

# Solve the differential equations
t_span = [0, 50]
t_eval = np.linspace(0, 50, 1001)
sol = solve_ivp(eqs, t_span, P_0, method='RK45', t_eval=t_eval)
P = sol.y

# Plot results
fig, axs = plt.subplots(4, 2, figsize=(7, 7))
fig.suptitle("Plots")
plt.subplots_adjust(top=0.9)  # Prevent layout warning

labels = ["$x_{1}$", "$u_{1}$", "$x_{2}$", "$u_{2}$", "$y_{1}$", "$w_{1}$", "$y_{2}$", "$w_{2}$"]
for i, ax in enumerate(axs.flat):
    ax.plot(t_eval, P[i])
    ax.set_ylabel(labels[i])
    if i >= 6:
        ax.set_xlabel("$t$")

# Animation
frames_per_second = 20
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.scatter(P[1, i], P[5, i], label='Mass 1')
    ax.scatter(P[3, i] + l0, P[7, i], label='Mass 2')
    ax.set_xlim(-l0, 2 * l0)
    ax.set_ylim(-l0, l0)
    ax.legend()

animate = FuncAnimation(fig, animate, frames=100, interval=1000//frames_per_second)
animate.save('my-anime.gif', fps=frames_per_second, dpi=100)
