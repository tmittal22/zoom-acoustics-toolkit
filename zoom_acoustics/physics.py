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


# ---------------------------------------------------------------- glide interpretation
C_W = 1482.0          # m/s, sound speed in water at 20 C


def wood_sound_speed(beta, kappa=GAMMA_CO2, P0=P_ATM, rho_w=RHO_W, c_w=C_W):
    """Wood's-law (low-frequency) sound speed [m/s] of a bubbly liquid, void fraction beta:

        1 / (rho_m c_m^2) = (1 - beta)/(rho_w c_w^2) + beta/(kappa P0),   rho_m ~ (1-beta) rho_w

    The gas makes the mixture far more compressible while barely changing its density, so
    c_m collapses: 1482 m/s (beta = 0) -> ~909 m/s (1e-4) -> ~354 m/s (1e-3) -> ~115 m/s
    (1e-2).  Valid well BELOW the individual bubbles' resonance; Commander & Prosperetti
    (1989, JASA 85) show the linear model holds to ~1-2 % void fraction.  kappa = 1.304
    is the adiabatic limit; small bubbles at low frequency are closer to isothermal
    (kappa -> 1), which lowers c_m by up to sqrt(1.304) = 1.14x."""
    beta = np.asarray(beta, float)
    compl = (1.0 - beta) / (rho_w * c_w ** 2) + beta / (kappa * P0)
    return 1.0 / np.sqrt((1.0 - beta) * rho_w * compl)


def layer_mode(beta, depth_m, mode="quarter", **kw):
    """Lowest mode [Hz] of a bubbly liquid layer of thickness depth_m:
    quarter-wave (rigid bottom, free top)  f = c_m / (4 h);  half-wave  f = c_m / (2 h).
    As gas accumulates (beta up), c_m and hence f go DOWN: one mechanism for a glide."""
    c = wood_sound_speed(beta, **kw)
    return c / ((4.0 if mode == "quarter" else 2.0) * np.asarray(depth_m, float))


def beta_from_layer_mode(f_hz, depth_m, mode="quarter", beta_max=0.02, **kw):
    """Invert layer_mode for the void fraction [-] given an observed frequency and an
    ASSUMED layer thickness (bisection in log beta).  NaN where no beta <= beta_max (the
    stated validity limit of the linear model) reproduces f, i.e. the layer would have to be
    a foam; NaN also where f exceeds the bubble-free mode c_w/(4h)."""
    f = np.atleast_1d(np.asarray(f_hz, float))
    c_target = f * (4.0 if mode == "quarter" else 2.0) * np.asarray(depth_m, float)
    lo = np.full_like(c_target, 1e-12)
    hi = np.full_like(c_target, beta_max)
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        fast = wood_sound_speed(mid, **kw) > c_target
        lo = np.where(fast, mid, lo)
        hi = np.where(fast, hi, mid)
    b = np.sqrt(lo * hi)
    bad = (b > 0.98 * beta_max) | (c_target >= wood_sound_speed(1e-12, **kw))
    b = np.where(bad, np.nan, b)
    return b[0] if b.size == 1 else b


def wall_factor(R, h):
    """Frequency factor f_wall / f_free for a bubble of radius R whose centre is a distance
    h from a RIGID plane wall, from the image-source correction to the radiation mass
    (the effect treated by Strasberg 1953, JASA 25):  (1 + R / (2 h))^(-1/2).
    Touching the wall (h = R): 0.816.  A wall LOWERS the frequency, so a Minnaert radius read
    from an attached bubble is too large by up to 1/0.816 = 1.22x."""
    return (1.0 + np.asarray(R, float) / (2.0 * np.asarray(h, float))) ** -0.5


def fritz_departure_radius(contact_diameter_m, sigma=SIGMA_W, rho=RHO_W, g=9.81):
    """Quasi-static departure radius of a bubble held by surface tension on a site of
    contact diameter d:  (4/3) pi R^3 rho g = pi d sigma  ->  R = (3 d sigma / (4 rho g))^(1/3).
    An UPPER bound on departure size: any flow sweeps bubbles off earlier."""
    return (3.0 * np.asarray(contact_diameter_m, float) * sigma / (4.0 * rho * g)) ** (1.0 / 3.0)
