"""
UE Power Consumption Model (3GPP TR 38.840).

Implements the full component model:
    P_UE = P_PA + P_BB + P_RF + P_WUR + P_static

All values in milliwatts (mW). Parameters traceable to 3GPP TR 38.840
and published 5G UE power measurements (Qualcomm, Ericsson whitepapers).
"""

import numpy as np
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Hardware parameters (mW unless noted)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UEHardwareProfile:
    # Power Amplifier
    pa_full_mw:     float = 520.0    # PA at full BW, max Tx power 26 dBm
    pa_efficiency:  float = 0.35     # drain efficiency full BW
    pa_lite_factor: float = 0.16     # beta_PA: kappa * eta_full/eta_lite

    # Baseband
    bb_static_mw:   float = 30.0     # P_BB^0 (static baseband floor)
    bb_full_mw:     float = 180.0    # P_BB at full BW
    bb_lite_kappa:  float = 0.20     # BWP-lite BW ratio (20 MHz / 100 MHz)

    # RF chains (4T8R device)
    n_chains:       int   = 4
    chain_mw:       float = 52.0     # per RF chain

    # WUR
    wur_rx_mw:      float = 2.0      # WUR actively receiving
    wur_sleep_mw:   float = 0.1      # WUR in sleep (monitoring only)

    # Static / misc
    static_mw:      float = 65.0     # CPU, memory, display baseline

    # Switching cost
    switch_energy_mj: float = 0.3    # mJ per BWP switch (RRC overhead)


# Default profile (mid-range 5G/6G smartphone)
DEFAULT_PROFILE = UEHardwareProfile()


class UEPowerModel:
    """
    Compute instantaneous UE power consumption given radio state.

    Parameters
    ----------
    profile : UEHardwareProfile
    """

    def __init__(self, profile: UEHardwareProfile = DEFAULT_PROFILE):
        self.p = profile

    def compute(self,
                bwp_full:    bool  = True,
                wur_active:  bool  = False,
                n_sym_active: int  = 14,
                prev_bwp_full: bool = True,
                slot_ms: float = 1.0) -> dict:
        """
        Compute power breakdown for one 1ms slot.

        Parameters
        ----------
        bwp_full      : True = full BW, False = BWP-lite
        wur_active    : True if WUR is in active receive mode
        n_sym_active  : OFDM symbols active this slot (0=sleep, 14=full)
        prev_bwp_full : BWP mode in previous slot (for switching cost)
        slot_ms       : slot duration in ms

        Returns
        -------
        dict with keys: total, pa, bb, rf, wur, static, switch
        """
        p = self.p
        activity = n_sym_active / 14.0   # 0..1

        # --- PA ---
        if activity > 0:
            if bwp_full:
                p_pa = p.pa_full_mw * activity
            else:
                p_pa = p.pa_full_mw * p.pa_lite_factor * activity
        else:
            p_pa = 0.0

        # --- BB ---
        if activity > 0:
            if bwp_full:
                p_bb = p.bb_static_mw + (p.bb_full_mw - p.bb_static_mw) * activity
            else:
                kappa = p.bb_lite_kappa
                p_bb_full_active = p.bb_static_mw + (p.bb_full_mw - p.bb_static_mw) * activity
                p_bb = p.bb_static_mw + kappa * (p_bb_full_active - p.bb_static_mw)
        else:
            p_bb = p.bb_static_mw * 0.1   # deep sleep: BB mostly off

        # --- RF chains ---
        if activity > 0:
            if bwp_full:
                p_rf = p.n_chains * p.chain_mw
            else:
                p_rf = p.chain_mw   # single chain in BWP-lite
        else:
            p_rf = 0.0

        # --- WUR ---
        p_wur = p.wur_rx_mw if wur_active else p.wur_sleep_mw

        # --- Static ---
        p_static = p.static_mw

        # --- Switching cost (amortised over slot) ---
        switched = (bwp_full != prev_bwp_full)
        p_switch = (p.switch_energy_mj / slot_ms) if switched else 0.0

        total = p_pa + p_bb + p_rf + p_wur + p_static + p_switch

        return {
            "total":  total,
            "pa":     p_pa,
            "bb":     p_bb,
            "rf":     p_rf,
            "wur":    p_wur,
            "static": p_static,
            "switch": p_switch,
        }

    def ssb_overhead_mw(self, ssb_period_ms: int) -> float:
        """
        Average additional power from periodic SSB decoding.
        SSB burst duration = 0.14 ms, decode power ~120 mW.
        """
        t_ssb_ms    = 0.14
        p_decode_mw = 120.0
        return (t_ssb_ms / ssb_period_ms) * p_decode_mw

    def power_floor_mw(self) -> float:
        """Minimum achievable power (BWP-lite + WUR sleep + static)."""
        return (self.p.bb_static_mw * 0.1
                + self.p.wur_sleep_mw
                + self.p.static_mw)

    def power_always_on_mw(self) -> float:
        """Always-on baseline power."""
        return (self.p.pa_full_mw
                + self.p.bb_full_mw
                + self.p.n_chains * self.p.chain_mw
                + self.p.wur_rx_mw
                + self.p.static_mw)


if __name__ == "__main__":
    model = UEPowerModel()
    print("=== Power breakdown examples ===")
    cases = [
        ("Always-on, full BW",   dict(bwp_full=True,  wur_active=True,  n_sym_active=14)),
        ("BWP-lite, WUR active", dict(bwp_full=False, wur_active=True,  n_sym_active=7)),
        ("Deep sleep, WUR only", dict(bwp_full=False, wur_active=False, n_sym_active=0)),
        ("BWP switch penalty",   dict(bwp_full=True,  wur_active=False, n_sym_active=14,
                                       prev_bwp_full=False)),
    ]
    for label, kwargs in cases:
        out = model.compute(**kwargs)
        print(f"  {label:<30s}: {out['total']:6.1f} mW "
              f"(PA={out['pa']:.0f} BB={out['bb']:.0f} "
              f"RF={out['rf']:.0f} WUR={out['wur']:.1f})")

    print(f"\nPower floor:    {model.power_floor_mw():.1f} mW")
    print(f"Always-on:      {model.power_always_on_mw():.1f} mW")
    for p in [5, 10, 20, 40]:
        print(f"SSB overhead @ {p:2d} ms: {model.ssb_overhead_mw(p):.2f} mW")
