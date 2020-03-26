#    Copyright 2019 André Melo, Sebastian Rubbert, and Anton R. Akhmerov, see LICENSE.txt

#    Contains modified code from https://zenodo.org/record/2578027#.XMrsZhexVhG
#    Copyright 2019 Bas Nijholt, Tom Laeven, Michael Wimmer, and Anton R. Akhmerov
#
#    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import cmath
import itertools
import functools
import operator
from math import cos, sin

import kwant
import kwant.continuum
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.constants
import numpy as np


sigma_0 = np.eye(2)
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
constants = dict(
    m_eff=0.04 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    exp=cmath.exp,
    cos=cmath.cos,
    sin=cmath.sin)


class Shape:
    """Creates callable object serving as a 'shape' function
    to be used with `kwant.Builder`.

    This class supports multiple set operations. Let `s₁` and `s₂` be
    instances of the `Shape` object, then:
    s₃ = s₁*s₂ is the intersection of the two shapes, meaning that s₃(x)
        returns True if and only if both s₁(x) and s₂(x) return True.
    s₃ = s₁+s₂ is the union of the two shapes, meaning that s₃(x) returns
        True if and only if s₁(x) and/or s₂(x) returns True.
    s₃ = s₁-s₂ is the difference of the two shapes, meaning that s₃(x) returns
        True if and only if s₁(x) returns True and s₂(x) returns False.

    The Shape class also contains some other useful methods to obtain the
    inverse, the edge, and interior of the shape.

    Using slice indexing one creates rectangular bounds, i.e.:
    if s = Shape()[:0, :],
    then s(x, y) returns True if and only if x<=0.

    Parameters
    ----------
    shape : callable, optional
        shape(kwant.Site) -> bool (default always returns True).
    """

    def __init__(self, shape=None):
        self.shape = shape if shape is not None else lambda site: True
        prod = itertools.product([-1, 0, 1], repeat=2)
        self._directions = [tup for tup in prod if tup != (0, 0)]

    def __call__(self, site):
        return self.shape(site)

    def __add__(self, other_shape):
        shapes = (self.shape, other_shape.shape)
        def union(site): return any(shape(site) for shape in shapes)
        return Shape(union)

    def __sub__(self, other_shape):
        shape_A, shape_B = (self.shape, other_shape.shape)
        def difference(site): return shape_A(site) and not shape_B(site)
        return Shape(difference)

    def __mul__(self, other_shape):
        shapes = (self.shape, other_shape.shape)
        def intersection(site): return all(shape(site) for shape in shapes)
        return Shape(intersection)

    @staticmethod
    def slice_func(_slice):
        def _func(x):
            y = True
            if _slice.start is not None:
                y &= _slice.start <= x
            if _slice.stop is not None:
                y &= x < _slice.stop
            return y
        return _func

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            # Make sure that item is iterable
            item = (item,)
        shapes = [Shape(lambda s, i=i, x=x: self.slice_func(x)(s.pos[i]))
                  for i, x in enumerate(item)]
        return functools.reduce(operator.mul, [self] + shapes)

    def inverse(self):
        """Returns inverse of shape."""

        return Shape(lambda site: not self.shape(site))

    def edge(self, which='inner'):
        """Returns edge of a 2D shape.

        If which is 'inner' ('outer'), the inner (outer) edge of the shape
        is returned, meaning the familiy of sites which lie inside (outside)
        the shape, but have at least one neighbor outside (inside) of the shape.
        """
        def edge_shape(site):
            def in_shape(x): return self.shape(Site(site.family, site.tag + x))
            sites = [in_shape(x) for x in self._directions]
            if which == 'inner':
                return self.shape(site) and not all(sites)
            elif which == 'outer':
                return not self.shape(site) and any(sites)
        return Shape(edge_shape)

    def interior(self):
        """Returns shape minus its edge."""
        return Shape(self - self.edge('inner'))

    @classmethod
    def below_curve(cls, curve):
        """Returns instance of Shape which returns True if a site (x, y)
        is such that y < curve(x)."""
        def _shape(site):
            x, y = site.pos
            return y < curve(x)
        return Shape(_shape)

    @classmethod
    def above_curve(cls, curve):
        """Returns instance of Shape which returns True if a
        site (x, y) is such that y >= curve(x)."""
        return Shape.below_curve(curve).inverse()

    @classmethod
    def left_of_curve(cls, curve):
        """Returns instance of Shape which returns True if a
        site (x, y) is such that x < curve(y)."""
        def _shape(site):
            x, y = site.pos
            return x < curve(y)
        return Shape(_shape)

    @classmethod
    def right_of_curve(cls, curve):
        """Returns instance of Shape which returns True if a site (x, y)
        is such that x >= curve(y)."""
        return Shape.left_of_curve(curve).inverse()


@functools.lru_cache()
def get_template_strings(tranverse_SOI=True):
    kinetic = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2) - mu {}) * kron(sigma_0, sigma_z)"
    kinetic = kinetic.format("+ m_eff*alpha^2 / (2 * hbar^2)")
    spin_orbit = "- alpha * kron(sigma_y, sigma_z) * k_x"
    if tranverse_SOI:
        spin_orbit += "+ alpha * kron(sigma_x, sigma_z) * k_y"
    superconductivity = "+ Delta * (cos({0}2 * pi * x / lbd_{1}) * kron(sigma_0, sigma_x) \
                          + sin({0}2 * pi * x / lbd_{1}) * kron(sigma_0, sigma_y))"
    superconductivity_mid = "+ Delta_middle * kron(sigma_0, sigma_x)"
    ham_periodic_pot = "+ V * cos(2 * pi * x / lbd_V) * kron(sigma_0, sigma_z)"

    ham_normal = kinetic + spin_orbit + ham_periodic_pot
    ham_sc_top = kinetic + \
        superconductivity.format('-', 'top') + ham_periodic_pot
    ham_sc_bot = kinetic + \
        superconductivity.format('+', 'bot') + ham_periodic_pot
    ham_sc_mid = kinetic + superconductivity_mid + ham_periodic_pot

    template_strings = {"normal": ham_normal,
                        "sc_top": ham_sc_top,
                        "sc_bot": ham_sc_bot,
                        "sc_mid": ham_sc_mid}

    return template_strings


def get_zigzag_shape(L_x, W, L_sc_top, L_sc_bot, w, z_x, z_y, a):
    def curve(x):
        x += z_x / 4
        if x % z_x < z_x / 2:
            return 4 * z_y / z_x * (x % (z_x / 2)) - z_y
        else:
            return -4 * z_y / z_x * (x % (z_x / 2)) + z_y

    y_offset = W / np.cos(np.arctan(4 * z_y / z_x)) if z_y != 0 else W

    _below_shape = Shape.below_curve(lambda x: curve(x) + y_offset // 2)
    _above_shape = Shape.above_curve(lambda x: curve(x) - y_offset // 2)
    _middle_shape = (_below_shape * _above_shape)[0:L_x + a, :]

    sc_top_initial_site = (0, y_offset // 2 + a)
    sc_top_shape = _middle_shape.inverse(
    )[0:L_x + a, :L_sc_top + y_offset // 2 + z_y]
    sc_bot_initial_site = (0, -y_offset//2-a)
    sc_bot_shape = _middle_shape.inverse(
    )[0:L_x + a, -L_sc_bot - y_offset // 2 - z_y:]

    if w != 0:
        if z_y != 0:
            raise NotImplementedError(
                'Can only add superconductor in straight junction geometry.')
        elif (w / a) % 2 != 1:
            raise RuntimeError('w / a must be an odd number.')

        _below_shape = Shape.below_curve(lambda x: w / 2)
        _above_shape = Shape.above_curve(lambda x: -w / 2)
        sc_mid_shape = (_below_shape * _above_shape)[0:L_x + a, :]
        sc_mid_initial_site = (0, 0)
        interior_shape = _middle_shape - sc_mid_shape

        return {'sc_top': (sc_top_shape, sc_top_initial_site),
                'sc_mid': (sc_mid_shape, sc_mid_initial_site),
                'sc_bot': (sc_bot_shape, sc_bot_initial_site),
                'normal_bot': (interior_shape, (a, -w / 2 - a)),
                'normal_top': (interior_shape, (a, w / 2 + a))}

    else:
        interior_shape = _middle_shape

        return {'sc_top': (sc_top_shape, sc_top_initial_site),
                'sc_bot': (sc_bot_shape, sc_bot_initial_site),
                'normal': (interior_shape, (a, a))}


@functools.lru_cache()
def system(L_x, W, L_sc_top, L_sc_bot, w, z_x, z_y, a, periodic, leads=False, transverse_SOI=True):
    """Create Phase Majorana system

    Parameters
    ----------
    L_x : float
        Length of the system (x-dimension=.
    W : float
        Width of the semiconductor (or contact separation of the junction.)
    L_sc_top : float
        Minimum width of the top superconductor.
    L_sc_bot : float
        Minimum width of the bottom superconductor.
    w : float
        Width of the middle superconductor.
    z_x : float
        Period of the zigzag.
    z_y : float
        Amplitude of the zigzag.
    a : float
        Lattice spacing.
    periodic : bool
        Toggle a wraparound system, such that the translational invariance
        is transformed into the momentum parameter k_x.
    leads : bool
        Add horizontal normal leads to the system.
    transverse_SOI: bool
        Add spin-orbit coupling in the y direction.

    Returns
    -------
    kwant.builder.FiniteSystem (periodic = True) or
    kwant.builder.InfiniteSystem (periodic = False).

    """
    #   If the system is periodic shorten the length by one lattice constant
    if periodic:
        L_x = L_x - a

    template_strings = get_template_strings(transverse_SOI)
    templates = {k: kwant.continuum.discretize(v, coords=('x', 'y'), grid=a)
                 for k, v in template_strings.items()}
    shapes = get_zigzag_shape(L_x, W, L_sc_top, L_sc_bot, w, z_x, z_y, a)

    if periodic:
        syst = kwant.Builder(kwant.TranslationalSymmetry([L_x + a, 0]))
    else:
        syst = kwant.Builder()

    if w == 0:
        normal_sites = syst.fill(templates['normal'], *shapes['normal'])

    else:
        norm_top_sites = syst.fill(templates['normal'], *shapes['normal_top'])
        norm_bot_sites = syst.fill(templates['normal'], *shapes['normal_bot'])
        sc_mid_sites = syst.fill(templates['sc_mid'], *shapes['sc_mid'])

    if L_sc_top > 0:
        sc_top_sites = syst.fill(templates['sc_top'], *shapes['sc_top'])

    if L_sc_bot > 0:
        sc_bot_sites = syst.fill(templates['sc_bot'], *shapes['sc_bot'])

    if periodic:
        syst = kwant.wraparound.wraparound(syst)

    if leads:
        if z_x != 0 and L_x % z_x != 0:
            raise NotImplementedError(
                'Horizontal leads for L_x not and integer multiple of z_x are not implemented.', z_x, L_x)

        ph = np.kron(sigma_y, sigma_y)
        c_law = np.kron(sigma_0, sigma_z)

        lead_left = kwant.Builder(kwant.TranslationalSymmetry(
            [-a, 0]), conservation_law=c_law, particle_hole=ph)
        lead_right = kwant.Builder(kwant.TranslationalSymmetry(
            [a, 0]), conservation_law=c_law, particle_hole=ph)

        #   Can't use lead.reversed() because the system might not be reflection
        #   invariant if it has a zigzag shape
        for lead in [lead_left, lead_right]:
            lead_idx = 0 if lead == lead_left else -1
            x_lead = 0 if lead == lead_left else L_x

            lead_shape = shapes['normal_bot'][0] + shapes['normal_top'][0] + \
                shapes['sc_mid'][0] if w != 0 else shapes['normal'][0]
            lead_shape = (lead_shape[lead_idx:, ::], (x_lead, 0))
            lead.fill(templates['normal'], *lead_shape)
            syst.attach_lead(lead)

    return syst.finalized()


def mumps_eigsh(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):

        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering='pord')
            instance.factor(matrix)
            self.solve = instance.solve
            sla.LinearOperator.__init__(self, matrix.dtype, matrix.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * sp.identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def spectrum(syst, params, k=20):
    """Compute the k smallest magnitude eigenvalues and
    corresponding eigenvectors of a system.

    Parameters
    ----------
    syst : `kwant.FiniteSystem` or `kwant.InfiniteSystem`
        System of which the spectrum will be computed.
    params : dict
        Dictionary containing the system parameters.
    k : int
        Number of eigenvalues to calculate.

    Returns
    -------
    energies : numpy array
        Array containing smallest magnitude eigenvalues.
    wfs : numpy array
        Array containing eigenvectors corresponding to eigenvalues.
    """
    ham = syst.hamiltonian_submatrix(params=params, sparse=True)
    (energies, wfs) = mumps_eigsh(ham, k=k, sigma=0)
    return energies, wfs


def scattering_invariant(syst, params):
    """Compute the scattering ivnariant Q = det r

    Parameters
    ----------
    syst : `kwant.FiniteSystem` or `kwant.InfiniteSystem`
        System of which the spectrum will be computed.
    params : dict
        Dictionary containing the system parameters.

    Returns
    -------
    invariant: numpy array
        Array containing smallest magnitude eigenvalues.
    """
    smatrix = kwant.smatrix(syst, 0.0, params=params)
    return np.linalg.det(smatrix.submatrix(0, 0))


def gap_from_transmission(syst, params, energy_vals, threshold):
    smatrices, transmissions = [], []
    for e in energy_vals:
        smatrix = kwant.smatrix(syst, e, params=params)
        smatrices.append(smatrix)
        transmissions.append(smatrix.transmission(0, 1))
    transmissions = np.array(transmissions)
    if np.any(transmissions > threshold):
        return energy_vals[np.argmax(transmissions > threshold)], (smatrices, transmissions)
    return np.max(energy_vals), (smatrices, transmissions)


@functools.lru_cache()
def get_site_by_y(syst, y):
    return {_y: [syst.id_by_site[site]
                 for site in syst.sites if site.pos[1] == _y]
            for _y in y}


def O(k_x, lbd, psi, syst, num_moments=5):
    """Compute the expectation value of the charge-momentum parity operator
    Any wavefunction ψ of the system may be expanded in the form

    ψ(x, y) = ∑_{i, k} c_{ik} f_{ik}(y) e^(ikx) / √L_x  |i⟩

    where i sums over spin and electron-hole orbitals and k is the wave
    vector  along the x direction.

    We multiply by e^(-ikx) / √L_x, integrate over x and obtain

    ψ_k(y) = ∫ dx  e^(-ikx) / √L_x ψ(x,y) = ∑_{i} c_{ik} f_{ik}(y) |i⟩

    Then we compute the moments

    ⟨O_k⟩ = (-1)^{n(k)} ∫ dy ψ_k(y)* 𝜏_z ψ_k(y)

    And finally the expectation value is given by

    ⟨O⟩ = ∑_{k} ⟨O_k⟩

    Parameters
    ----------
    k_x : float
        Crystal momentum (in units of 1 / lbd, running from -pi to pi).
    lbd : float
        Period of the superconducting phase. Assumed to be periodicity of
        the system.
    psi: numpy.array
        State for which to calculate the expectation value.
    num_moments: int
        Number of O_k to sum over.

    Returns
    -------
    O : numpy.complex128
        Expectation value of charge-momentum parity

    """
    tau_z = np.kron(np.eye(2), sigma_z)
    x = sorted(set([site.pos[0] for site in syst.sites]))
    y = sorted(set([site.pos[1] for site in syst.sites]))
    x, y = np.array(x), tuple(y)
    site_by_y = get_site_by_y(syst, y)

    #    k values for which we will compute moments
    bz_vals = np.arange(-num_moments, num_moments+1)
    k_vals = k_x/lbd + bz_vals * 2 * np.pi / lbd

    #    Assemble matrix with momentum wavefunctions
    k_wfs = [np.exp(1j * k * x) / np.sqrt(len(x)) for k in k_vals]
    k_wfs = np.array(k_wfs)

    #   Reshape psi to form psi_y,x,i where x, y denote coordinates
    #   and i runs over orbitals
    num_sites = len(x) * len(y)
    num_orbs = psi.shape[0] // num_sites
    psi = psi.reshape(num_sites, num_orbs)
    psi = [psi[site_by_y[_y]] for _y in y]

    #   Reorder tensor to form psi_x,y,i
    psi = np.swapaxes(psi, 0, 1)

    #   Integrate over x
    psi = np.tensordot(k_wfs.conj(), psi, axes=(1, 0))

    #   Compute O_k
    def n(k):
        abs_n = np.round(np.abs(k) / (2 * np.pi / lbd))
        return np.sign(k) * abs_n

    O = np.einsum('k,ii,kyi,kyi',
                  (-1)**n(k_vals), tau_z, psi.conj(), psi)

    return O


def best_match(psi1, psi2, threshold=None):
    """Find the best match of two sets of eigenvectors.

    Parameters
    ----------
    psi1, psi2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

    Returns
    -------
    sorting : numpy 1D integer array
        Permutation to apply to ``psi2`` to make the optimal match.
    diconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    if threshold is None:
        threshold = (2 * psi1.shape[0])**-0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = so.linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold
