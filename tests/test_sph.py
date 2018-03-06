"""
Tests for the SPH portion of the code.
"""
import matplotlib
matplotlib.use("Agg")

import seagen.sph as sph
import matplotlib.pyplot as plt
import numpy as np

def test_sph_linear_particles():
    eta = 1.0
    x = np.arange(1000)
    y = np.zeros(1000)
    z = np.zeros(1000)
    m = np.ones(1000)
    hsml_init = 2
    
    eos = sph.EquationOfState
    kernel = sph.cubic_spline
    
    SPHD = sph.SPHDataset(
        x, y, z, m, eta, hsml_init, kernel, eos
    )
    
    fig, axes = plt.subplots(1, 2)
    
    axes[0].plot(x, SPHD.hsml)
    axes[1].plot(x, SPHD.densities)
    
    axes[1].set_xlabel("$x$ position")
    axes[1].set_ylabel("Density")
    axes[0].set_ylabel("Smoothing length, $h$")
    
    plt.savefig("test_sph_linear_particles.png")
    
    
def test_sph_hsml_grad_view():
    eta = 1.0
    x = np.arange(1000)
    y = np.zeros(1000)
    z = np.zeros(1000)
    m = np.ones(1000)
    hsml_init = 0.5
    
    eos = sph.EquationOfState
    kernel = sph.cubic_spline
    
    SPHD = sph.SPHDataset(
        x, y, z, m, eta, hsml_init, kernel, eos
    )
    
    y = [SPHD.constraint_equation(np.ones(1000) * h).sum() for h in np.arange(0, 3, 0.1)]
    
    fig, axes = plt.subplots(1, 1)
    
    axes.plot(np.arange(0, 3, 0.1), y)
    axes.set_xlabel("Smoothing Length")
    axes.set_ylabel("Diff")
   
    plt.savefig("test_sph_hsml_grad_view.png")
    
if __name__ == "__main__":
    test_sph_linear_particles()
    test_sph_hsml_grad_view()