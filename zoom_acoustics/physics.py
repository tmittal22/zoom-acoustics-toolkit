"""
physics.py -- bubble acoustics helpers (SI units).  Ported from the Sep-2026 analysis4
pipeline (a4_phys.py), where they were validated against closed-form limits.

READ docs/STUDENT_GUIDE.md section "Bubble size" before converting any frequency to a
radius.  In the Sep-2026 data, per-event peak-frequency sizing was FALSIFIED on three of
four channels: a 5.6 kHz apparatus resonance and strongly over-damped ringdowns (Q ~ 6
measured vs ~31 predicted) mean a spectral peak is not a free bubble.  The band -> radius
labels below are a guide to which bubble sizes COULD radiate in a band, not a measurement.
"""
from __future__ import annotations

import numpy as np

P_ATM = 101325.0      # Pa
RHO_W = 998.0         # kg/m^3, water at 20 C
SIGMA_W = 0.0728      # N/m, water/air at 20 C
GAMMA_CO2 = 1.304     # adiabatic index of CO2 (adiabatic limit)


def minnaert_f0(R, kappa=GAMMA_CO2, P0=P_ATM, rho=RHO_W, sigma=SIGMA_W):
    """Breathing-mode resonance [Hz] of a gas bubble of radius R [m]:

        f0 = sqrt(3 kappa (P0 + 2 sigma/R) - 2 sigma/R) / (2 pi R sqrt(rho))

    Large-R limit: f0 R -> sqrt(3 kappa P0 / rho) / 2 pi = 3.17 m Hz for CO2 in water.
    """
    R = np.asarray(R, float)
    return np.sqrt(3 * kappa * (P0 + 2 * sigma / R) - 2 * sigma / R) / (2 * np.pi * R * np.sqrt(rho))


def minnaert_R(f0, **kw):
    """Invert minnaert_f0 for R [m] by bisection in log R (200 steps; exact to ~1e-15)."""
    f0 = np.atleast_1d(np.asarray(f0, float))
    lo = np.full_like(f0, 1e-7)
    hi = np.full_like(f0, 1e-1)
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        big = minnaert_f0(mid, **kw) > f0
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    r = np.sqrt(lo * hi)
    return r[0] if r.size == 1 else r


def band_radius_label(lo_hz, hi_hz):
    """'R 0.13-2.6 mm' style label: the Minnaert radius range of a frequency band."""
    if lo_hz <= 0:
        return ""
    r_hi, r_lo = minnaert_R(lo_hz), minnaert_R(hi_hz)
    if r_hi >= 1e-3:
        return f"R {r_lo*1e3:.2g}-{r_hi*1e3:.2g} mm"
    return f"R {r_lo*1e6:.0f}-{r_hi*1e6:.0f} um"


def quarter_wave_mode(depth_m, c=1480.0):
    """Lowest vertical mode of a liquid layer over a rigid bottom, f = c / 4h [Hz].
    Real vessels shift this (wall compliance, meniscus): measure it with a water-only take."""
    return c / (4.0 * np.asarray(depth_m, float))


def capillary_length(sigma=SIGMA_W, rho=RHO_W, g=9.81):
    """l_c = sqrt(sigma / (rho g)) [m]; 2.73 mm for water at 20 C.  A gas bubble much larger
    than this is flattened by buoyancy and is not a free sphere."""
    return float(np.sqrt(sigma / (rho * g)))


def bond_number(R, sigma=SIGMA_W, rho=RHO_W, g=9.81):
    """Bo = rho g R^2 / sigma = (R / l_c)^2.  Bo >= 1: buoyancy beats surface tension, so a
    Minnaert (free spherical bubble) reading of that radius is not self-consistent."""
    return rho * g * np.asarray(R, float) ** 2 / sigma
