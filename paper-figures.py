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

# # Figures for "Supercurrent-induced Majorana bound states in a planar geometry".
#
# By André Melo, Sebastian Rubbert and Anton R. Akhmerov.
#
# If the `data` directory is populated, the notebook will skip calculations and simply produce the plots in the paper. To do the calculations from scratch simply delete the contents of the directory. However, running the code with default parameters in reasonable time will require a computational cluster.

# ## Set up computing environment

#   Taken from adaptive.notebook_integration.py
#   https://github.com/python-adaptive/adaptive/blob/master/adaptive/notebook_integration.py
def in_ipynb():
    try:
        # If we are running in IPython, then `get_ipython()` is always a global
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False

# +
import os
import adaptive
if in_ipynb():
    adaptive.notebook_extension()

use_cluster = False
engines = 100
work_folder = '/home/adealmeidanasc/Sync/phasemajorana'

if use_cluster:
    import hpc05
    client, dview, lview = hpc05.connect.start_remote_and_connect(engines,
                                                                  folder=work_folder)
    def map_fn(fn, iterable):
        r = lview.map_async(fn, iterable)
        r.wait_interactive()
        return r
else:
    client = None
    def map_fn(fn, iterable):
        return list(map(fn, iterable))

#   If not using the cluster, temporarily redefine %%px magic
#   to not do anything
if not use_cluster and in_ipynb():
    def px(line, cell):
        res = get_ipython().run_cell(cell)
    get_ipython().register_magic_function(px, 'cell')
# -

# ## Imports

# +
# %%px --local
import numpy as np
from scipy import linalg
import scipy.constants
import cmath
import pickle
from functools import partial

import kwant, phasemajoranas

# %load_ext autoreload
# %autoreload 2
# -

# ## Plotting options

# +
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import holoviews as hv
hv.extension('matplotlib')

plot_params = {
          'backend': 'ps',
          'axes.labelsize': 20,
          'font.size': 20,
          'legend.fontsize': 10,
          'xtick.labelsize': 20,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Roman',
          'legend.frameon': True,
          'savefig.dpi': 300,
         }

plt.rcParams.update(plot_params)
plt.rc('text.latex', preamble=[r'\usepackage{xfrac}'])
# -

# ## Default system parameters

# +
# %%px --local

constants = {
    'm_eff': 0.04 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,
    'hbar': scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    'e': scipy.constants.e,
    'exp': cmath.exp,
    'cos': cmath.cos,
    'sin': cmath.sin
}

params_raw = {
    'mu': 0.1,
    'alpha': 10,
    'Delta': 1,
    'Delta_middle': 0,
    'lbd_top': np.inf,
    'lbd_bot': np.inf,
    'V': 0,
    'lbd_V': np.inf
}

default_params = dict(**constants, **params_raw)

default_syst_pars = {
    'W': 150,
    'L_x': 370,
    'L_sc_top': 200,
    'L_sc_bot': 200,
    'w': 0,
    'z_x': 370,
    'z_y': 0,
    'a': 10,
    'periodic': True,
    'leads': False
}
# -

# ## Plot band structures

# +
# %%px --local

def bands(k_x, mu=0.09, lbd_top=370, lbd_bot=370, V=0, num_bands=8,
          transverse_SOI=False):
    L_x = max([lbd_top, lbd_bot])
    syst_pars = dict(default_syst_pars)
    syst_pars['L_x'] = L_x
    syst_pars['z_x'] = L_x
    syst_pars['transverse_SOI'] = transverse_SOI
    syst = phasemajoranas.system(**syst_pars)

    params = dict(default_params)
    params['mu'] = mu
    params['lbd_top'] = lbd_top
    params['lbd_bot'] = lbd_bot
    params['k_x'] = k_x
    params['V'] = V
    params['lbd_V'] = L_x

    energies, wfs = phasemajoranas.spectrum(syst, params, k=num_bands)
    O_vals = [phasemajoranas.O(k_x, L_x, wf, syst, num_moments=5)
              for wf in wfs.T]
    asrt = np.argsort(energies)
    return energies[asrt], wfs[:, asrt], np.array(O_vals)[asrt]

kvals = np.linspace(-np.pi, np.pi, 501)
param_keys = ['mu', 'lbd_top', 'lbd_bot', 'V', 'transverse_SOI']
param_vals = [
    [0.2, 370, 370, 0, False],         # Add supercurrent, no tr. SOI
    [0.2, 370, 370, 0, True],          # Add supercurrent, with tr. SOI
    [0.45, 700, 350, 0, True],         # Broken inversion symmetry
    [0.3, 370, 370, 0, True],          # Tune chemical potential
    [0.3, 370, 370, 0.005, True],      # Add potential
]
params = [dict(zip(param_keys, vals)) for vals in param_vals]
fns = [partial(bands, **p) for p in params]
# -

offset = default_params['m_eff'] * default_params['alpha']**2 / (2 * constants['hbar']**2)
print('μ values with constant offset:')
print([p[0] - offset for p in param_vals])

if os.path.exists('data/bandstructures.p'):
    results = pickle.load(open('data/bandstructures.p', 'rb'))
else:
    results = []
    for i, fn in enumerate(fns):
        r = map_fn(fn, kvals)
        energies, wfs, O = zip(*r)
        energies, wfs, O = np.array(energies), np.array(wfs), np.array(O)

        if i >= 3:
            e, psi = energies[0], wfs[0]
            srt_energies = [e]
            srt_O = [O[0]]

            for i in range(1, len(kvals)):
                k2, e2, psi2 = kvals[i], energies[i], wfs[i]
                perm, line_breaks = phasemajoranas.best_match(psi, psi2)
                e2 = e2[perm]
                psi = psi2[:, perm]
                e = e2
                srt_energies.append(e)
                srt_O.append(O[i][perm])
                
            results.append([np.array(srt_energies), np.real(srt_O)])
        else:
            results.append([energies, np.real(O)])

    pickle.dump(results, open('data/bandstructures.p', 'wb'))

# +
fig, ax = plt.subplots(1, 1)
plt.plot(kvals, results[0][0], c='C1', ls='--')
plt.plot(kvals, results[1][0], c='C0')

xvals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
xlabels = [r'$-\pi$', r'$\sfrac{-\pi}{2}$',
           r'$0$', r'$\sfrac{\pi}{2}$', r'$\pi$']
yvals = [-0.05, 0, 0.05]
ylabels = [f'${y}$' for y in yvals]

plt.xticks(xvals, xlabels)
plt.yticks(yvals, ylabels)
plt.xlim(-np.pi, np.pi)
plt.ylim([-0.06, 0.06])
plt.xlabel(r'$\lambda \kappa$')
plt.ylabel(r'$E / \Delta$', labelpad=-10)

plt.savefig('paper/figures/bstruct_effective_zeeman.pdf', bbox_inches="tight")
plt.show()
plt.close()

# +
fig, ax = plt.subplots(1, 1)
plt.plot(kvals, results[2][0], c='C0')

xvals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
xlabels = [r'$-\pi$', r'$\sfrac{-\pi}{2}$',
           r'$0$', r'$\sfrac{\pi}{2}$', r'$\pi$']
yvals = [-0.05, 0, 0.05]
ylabels = [f'${y}$' for y in yvals]

plt.xticks(xvals, xlabels)
plt.yticks(yvals, ylabels)
plt.xlim(-np.pi, np.pi)
plt.ylim([-0.06, 0.06])
plt.xlabel(r'$\lambda_T \kappa$')
plt.ylabel(r'$E / \Delta$', labelpad=-10)

plt.savefig('paper/figures/bstruct_inv_symm.pdf', bbox_inches="tight")
plt.show()
plt.close()

# +
cmap = matplotlib.cm.coolwarm

def plot_colourline(ax, x, y, c, lw=1.5):
    c = cmap((c+1)/2)
    for i in np.arange(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=c[i], lw=lw)

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 3.5))

for i, ax in zip(range(3, 5), axes):
    sorted_levels, sorted_O = results[i]
    for j in range(sorted_levels.shape[1]):
        plot_colourline(ax, kvals, sorted_levels[:, j], sorted_O[:, j])

yvals = [-0.05, 0, 0.05]
ylabels = [f'${y}$' for y in yvals]
xvals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
xlabels = [r'$-\pi$', r'$\sfrac{-\pi}{2}$',
           r'$0$', r'$\sfrac{\pi}{2}$', r'$\pi$']

for ax, label in zip(axes, ['(a)', '(b)']):
    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yvals)
    ax.set_yticklabels(ylabels)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-0.075, 0.075])
    ax.text(-np.pi * 0.96, 0.06, label, fontsize=18)
    ax.set_xlabel(r'$\lambda \kappa$')
    ax.set_xlabel(r'$\lambda \kappa$')

axes[0].set_ylabel(r'$E/ \Delta$', labelpad=-10)

#   Colobar
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for ax in axes:
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(r'$\left< \mathcal{O} \right>$', labelpad=5)
fig.delaxes(fig.axes[2])

plt.tight_layout(pad=0.1)
plt.savefig('paper/figures/bstruct_gap_open.pdf', bbox_inches="tight")
plt.show()
plt.close()
# -

# ## Plot phase diagrams

# +
# %%px --local

energy_vals = np.linspace(0, 0.05, 100)

diagram_bounds = [(0.8, 1.4), (400, 600)]

syst_pars = {
    'W': 150,
    'L_x': 515 * 20,
    'z_x': 515,
    'periodic': False,
    'leads': True
}
syst_pars = {**default_syst_pars, **syst_pars}
params = {**default_params, 'alpha': 20}

zigzag_syst_pars = {**syst_pars, **{'z_y': 37.5}}
zigzag_params = {**params}

pot_syst_pars = {**syst_pars}
pot_params = {**params, **{'V': 0.15, 'lbd_V': 515}}

sc_syst_pars = {**syst_pars, **{'w': 10}}
sc_params = {**params, **{'Delta_middle': 1}}

# +
# %%px --local

def gap_wrapper(mu_lbd, syst_pars=default_syst_pars,
                params=default_params, energy_vals=energy_vals,
                threshold=0.5):
    mu, lbd = mu_lbd
    _params = {**params, **{'mu': mu, 'lbd_top': lbd, 'lbd_bot': lbd}}
    syst = phasemajoranas.system(**syst_pars)
    invariant = np.real(phasemajoranas.scattering_invariant(syst, _params))
    gap, _ = phasemajoranas.gap_from_transmission(syst, _params, energy_vals,
                                                  threshold)

    return gap * invariant

gap_zigzag = partial(gap_wrapper,
                     syst_pars=zigzag_syst_pars,
                     params=zigzag_params)

gap_pot = partial(gap_wrapper,
                  syst_pars=pot_syst_pars,
                  params=pot_params)

gap_sc = partial(gap_wrapper,
                 syst_pars=sc_syst_pars,
                 params=sc_params)

# +
learners = [adaptive.Learner2D(fn, bounds=diagram_bounds)
            for fn in [gap_zigzag, gap_pot, gap_sc]]
cdims = [['System'], [('Zigzag',), ('Potential',), ('3rd superconductor',)]]
base_fname = 'data/'
fnames = [base_fname + fname 
          for fname in ['zigzag_gap.p', 'pot_gap.p', 'sc_gap.p']]

learner_gap = adaptive.BalancingLearner(learners,
                                        cdims=cdims,
                                        strategy='npoints')
learner_gap.load(fnames)

def goal(b_l):
    return np.all([l.npoints > 2000 for l in b_l.learners])

if not goal(learner_gap):
    runner = adaptive.Runner(learner_gap, goal, executor=client)
    runner.start_periodic_saving(dict(fname=fnames), interval=60)
    if in_ipynb():
        runner.live_info()

# +
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


#    We discard the last column of data to avoid NaN, see adaptive issue #181.
zigzag_data, pot_data, sc_data = map(lambda x: x.plot().Image.I.data[:, :-1],
                                     learner_gap.learners)
_cdata = np.append(np.append(zigzag_data, pot_data), sc_data)
_cdata = _cdata[~np.isnan(_cdata)]
vmax = np.max(_cdata)
vmin = np.min(_cdata)

#    Add offset to μ
offset = params['m_eff'] * params['alpha']**2 / (2 * constants['hbar']**2)
plot_bounds = np.reshape(diagram_bounds, 4)
plot_bounds[:2] -= offset

cmap_opts = {'cmap': 'RdBu_r',
             'norm': MidpointNormalize(midpoint=0.,
                                       vmin=vmin,
                                       vmax=vmax),
             'extent': plot_bounds,
             'aspect': 1/300,
             'interpolation': None}

fig, axes = plt.subplots(1, 3, sharey=True, tight_layout={'pad': 0.1})
cm_pot = axes[0].imshow(pot_data, **cmap_opts)
cm_zigzag = axes[1].imshow(zigzag_data, **cmap_opts)
cm_sc = axes[2].imshow(sc_data, **cmap_opts)

for i, (ax, label) in enumerate(zip(axes, ['(a)', '(b)', '(c)'])):
    ax.set_xlabel(rf'$\mu$ (meV)', fontsize=12)
    if i < 2:
        ax.axhline(y=515, ls='--', c='k', lw=0.8)
    xticks = np.arange(0.7, 1.5, 0.2)
    yticks = np.arange(400, 650, 50)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.1f' % t for t in xticks], fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%d' % t for t in yticks], fontsize=12)
    ax.text(0.2, 0.97, label, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            color='black', fontsize=12)

axes[0].set_ylabel(rf'$\lambda$ (nm)', fontsize=12)

for ax in axes[:2]:
    _cb = fig.colorbar(cm_pot, ax=ax, fraction=0.05)
    fig.delaxes(fig.axes[3])

cb = fig.colorbar(cm_zigzag, ax=axes[2], fraction=0.05)
cb.set_label(r'$E_{\mathrm{gap}} \cdot \mathcal{Q} / \Delta$', labelpad=5,
             fontsize=12)
cb.set_ticks(np.arange(-0.02, 0.04, 0.01))
cb.ax.tick_params(labelsize=12)

plt.savefig('paper/figures/phase_diagrams.pdf', bbox_inches="tight")
plt.show()
plt.close()
# -

# ### Appendix figures: single superconductor with zigzag

# +
# %%px --local
one_sc_diagram_bounds = [(1, 1.7), (250, 550)]

one_sc_syst_pars = {**syst_pars,
                    **{'L_sc_top': 0, 'z_x': 360, 'L_x': 360 * 20, 'z_y': 75}}
one_sc_params = {**params}
gap_one_sc = partial(gap_wrapper,
                     syst_pars=one_sc_syst_pars,
                     params=params)

# +
learner = adaptive.Learner2D(gap_one_sc, bounds=one_sc_diagram_bounds)
fname = 'data/one_sc_gap.p'
learner.load(fname)

def goal(l): return l.npoints > 2000

if not goal(learner):
    runner = adaptive.Runner(learner, goal, executor=client)
    runner.start_periodic_saving(dict(fname=fname), interval=60)
    if in_ipynb():
        runner.live_info()

# +
#    We discard the last column of data to avoid NaN, see adaptive issue #181.
one_sc_data = learner.plot().Image.I.data[:, :-1]
vmax = np.max(one_sc_data[~np.isnan(one_sc_data)])
vmin = np.min((one_sc_data[~np.isnan(one_sc_data)]))

#    Add offset to μ
plot_bounds = np.reshape(one_sc_diagram_bounds, 4)
plot_bounds[:2] -= offset

cmap_opts = {'cmap': 'RdBu_r',
             'norm': MidpointNormalize(midpoint=0.,
                                       vmin=vmin, 
                                       vmax=vmax),
             'extent': plot_bounds,
             'aspect': 1/450,
             'interpolation': None}

fig, ax = plt.subplots(1, 1, tight_layout={'pad': 0.1})

xticks = np.arange(1.1, 1.7, 0.2)
yticks = np.arange(250, 600, 50)
ax.set_xticks(xticks)
ax.set_xticklabels(['%.1f' % t for t in xticks])
ax.set_yticks(yticks)
ax.set_yticklabels([str(t) for t in yticks])
ax.tick_params(axis='y', which='major', pad=7)
ax.tick_params(axis='x', which='major', pad=7)

ax.set_xlabel(rf'$\mu$ (meV)')
ax.set_ylabel(rf'$\lambda$ (nm)')

cm_one_sc = ax.imshow(one_sc_data, **cmap_opts)
cb = fig.colorbar(cm_one_sc, fraction=0.05)
cb.set_label(r'$E_{\mathrm{gap}} \cdot \mathcal{Q} / \Delta$', labelpad=5, fontsize=20)

plt.xlim([1.0, 1.6])
plt.ylim([250, 500])

plt.savefig('paper/figures/phase_diagram_one_sc.pdf', bbox_inches="tight")
plt.show()
plt.close()
# -
# ## Transmission plot

# +
# %%px --local

def transmission(e, syst_pars=default_syst_pars, params=default_params):
    syst = phasemajoranas.system(**syst_pars)
    smatrix = kwant.smatrix(syst, e, params=params)
    return smatrix.transmission(0, 1)
# -

mu, lbd = 1.2, 420
cond_params = dict(sc_params, mu=mu, lbd_top=lbd, lbd_bot=lbd)
cond_syst_pars = dict(sc_syst_pars)
e = np.linspace(0, 0.03, 500)

if os.path.exists('data/transmission.p'):
    transmissions = pickle.load(open('data/transmission.p', 'rb'))
else:
    r = map_fn(
        partial(transmission, syst_pars=cond_syst_pars, params=cond_params), e)
    transmissions = np.array(list(r))
    pickle.dump(transmissions, open('data/transmission.p', 'wb'))

# +
e_gap = e[np.argmax(transmissions>0.5)]
t = transmissions[np.argmax(transmissions>0.5)]
ylim = 1.5

plt.plot(e, transmissions, c='black', lw=2)
plt.axvline(x=e_gap, ymax=0.5/ylim, c='r', lw=2, ls='--')
plt.axhline(y=0.5, xmax=e_gap/max(e), c='r', lw=2, ls='--')
plt.xlim([0, max(e)])
plt.ylim([-0.02, 1.5])
plt.xlabel(r'$E/\Delta$')
plt.ylabel(r'$T_{12}$')
plt.xticks([0, 0.01, 0.02, e_gap], ['0', '0.01', '0.02', r'$E_\text{gap}$'])
plt.savefig('paper/figures/transmission.pdf', bbox_inches="tight")
plt.show()
plt.close()
# -


