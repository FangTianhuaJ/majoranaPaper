# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python [conda env:dev]
#     language: python
#     name: conda-env-dev-py
# ---

# By André Melo, Sebastian Rubbert and Anton R. Akhmerov.
#
# Here we numerically verify what couplings are introduced by a periodic potential, a superconductor carrying no supercurent, and a zigzag-shaped junction.

# ## Imports

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
# %matplotlib inline

# ## Define parameters and operators

# +
#   Params
L_x = lbd = z_x = 25
L_m = 10
L_sc = 10
L_y = 2 * L_sc + L_m
z_y = 5
w = 2
Delta = Delta_middle = 1

#   Pauli matrices
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#   Spatial grid
x, y = np.arange(0, L_x), np.arange(0, L_y) - L_y / 2
xm, ym = np.meshgrid(x, y)
# -

# ##  Define shapes

# +
def zigzag_curve(x):
    x = x + z_x / 4
    if x % z_x < z_x / 2:
        return 4 * z_y / z_x * (x % (z_x / 2)) - z_y  
    return -4 * z_y / z_x * (x % (z_x / 2)) + z_y

def zigzag_shape(x, y):
    theta = np.arctan(4 * z_y / z_x)
    offset = L_m / np.cos(theta) / 2
    return (not zigzag_curve(x) - offset <= y <= zigzag_curve(x) + offset)

zigzag_shape = np.vectorize(zigzag_shape)

straight_junction = np.abs(ym) >= L_m / 2
mid_sc_junction = (np.abs(ym) <= w/2) 
zigzag_junction = zigzag_shape(xm, ym)
# -

fig, axes = plt.subplots(1, 3)
for ax, shape in zip(axes, [straight_junction, mid_sc_junction, zigzag_junction]):
    ax.imshow(shape)

# ## Make hamiltonians

# +
def sc_ham(delta, shape, phase):   
    phase = np.vectorize(phase)
    sc_re = delta * np.cos(phase(xm, ym)) * shape
    sc_im = delta * np.sin(phase(xm, ym)) * shape
    sc_re = np.diag(sc_re.reshape(L_x * L_y))
    sc_im = np.diag(sc_im.reshape(L_x * L_y))
    return np.kron(sc_re, tau_x) + np.kron(sc_im, tau_y)

def winding_phase(_x, _y):
    sign = 1 if _y > 0 else -1
    return sign * 2 * np.pi * _x / lbd

def constant_phase(_x, _y): 
    return 0

straight_ham = sc_ham(Delta, straight_junction, winding_phase)
mid_sc_ham = sc_ham(Delta_middle, mid_sc_junction, constant_phase) + straight_ham
zigzag_ham = sc_ham(Delta, zigzag_junction, winding_phase)
# -

# ## Plot hamiltonians in plane wave basis

# +
def k_wf(n, n_y=1):
    k_x = 2 * np.pi * n / L_x
    wf = np.exp(1j * k_x * xm) * np.sin(n_y * np.pi * (ym - L_y / 2) / L_y)
    wf = wf.reshape(L_x * L_y)
    return wf/np.linalg.norm(wf)

def k_e(n, n_y=1):
    return np.kron(k_wf(n, n_y), np.array([1, 0]))

def k_h(n, n_y=1): 
    return np.kron(k_wf(n, n_y), np.array([0, 1]))

n_vals = range(6)
U = np.array([k_e(n) for n in n_vals] + [k_h(n) for n in n_vals]).T
hams = [U.T.conj() @ ham @ U for ham in [straight_ham, mid_sc_ham, zigzag_ham]]

# +
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 4))
ticklabels = [r'$e_{%d}$' % n for n in n_vals] + [r'$h_%d$' % n for n in n_vals]
titles = ['Straight', 'Middle sc', 'Zigzag']

for ax, ham, title in zip(axes, hams, titles):
    im = ax.matshow(np.abs(ham))
    colorbar(im)
    ax.set_xticks(range(len(ticklabels)))
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(range(len(ticklabels)))
    ax.set_yticklabels(ticklabels)  
    dim = len(n_vals)
    ax.vlines(dim - 0.5, -0.5, 2 * dim - 0.5, colors='red')
    ax.hlines(dim - 0.5, -0.5, 2 * dim - 0.5, colors='red')
    ax.set_title(title, pad=25)

plt.tight_layout(h_pad=1)
# -


