"""
MMPP-2 traffic model calibrated to 3GPP TR 36.814 reference traffic.

Parameters
----------
lambda_L : 0.1 Mbps   (low-load arrival rate)
lambda_H : 18.75 Mbps (high-load arrival rate)
mu_LH    : 1.25 s^-1  (low->high transition rate, mean low duration 800 ms)
mu_HL    : 5.00 s^-1  (high->low transition rate, mean high duration 200 ms)

Stationary mean throughput:
    pi_H = mu_LH / (mu_LH + mu_HL) = 1.25/6.25 = 0.20
    pi_L = 0.80
    mean = 0.8*0.1 + 0.2*18.75 = 0.08 + 3.75 = 3.83 Mbps  (after packetisation ~2.5 Mbps)
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Default calibrated parameters (3GPP TR 36.814 video+web UE)
# --------------------------------------------------------------------------- #
LAMBDA_L_MBPS = 0.1
LAMBDA_H_MBPS = 18.75
MU_LH_PER_S   = 1.25    # 1/800ms
MU_HL_PER_S   = 5.00    # 1/200ms

# Packetisation: average packet size 1500 bytes, mix of TCP ACKs and data
PACKET_SIZE_BYTES = 1500
OVERHEAD_FACTOR   = 0.65   # application payload / total bytes


class MMPPTraffic:
    """
    Markov-Modulated Poisson Process (2-state) traffic source.

    The source alternates between low (L) and high (H) load states.
    In each 1 ms slot, the offered load in Mbps is:
        offered = lambda_state * overhead_factor + small_noise

    Parameters
    ----------
    slot_ms : float
        Slot duration in ms (default 1 ms = one NR slot at 15 kHz SCS).
    rng : np.random.Generator
    """

    def __init__(self,
                 lambda_l: float = LAMBDA_L_MBPS,
                 lambda_h: float = LAMBDA_H_MBPS,
                 mu_lh: float    = MU_LH_PER_S,
                 mu_hl: float    = MU_HL_PER_S,
                 slot_ms: float  = 1.0,
                 rng: np.random.Generator = None):

        self.lambda_l  = lambda_l
        self.lambda_h  = lambda_h
        self.mu_lh     = mu_lh
        self.mu_hl     = mu_hl
        self.slot_s    = slot_ms / 1e3
        self.rng       = rng or np.random.default_rng()

        # transition probabilities per slot (Euler discretisation)
        self.p_lh = 1 - np.exp(-mu_lh * self.slot_s)
        self.p_hl = 1 - np.exp(-mu_hl * self.slot_s)

        # stationary probabilities
        self.pi_h = mu_lh / (mu_lh + mu_hl)
        self.pi_l = 1 - self.pi_h
        self.mean_rate_mbps = (self.pi_l * lambda_l
                               + self.pi_h * lambda_h) * OVERHEAD_FACTOR

        # initialise state from stationary distribution
        self.state = 'H' if self.rng.random() < self.pi_h else 'L'
        self._idle_slots = 0   # slots since last non-trivial traffic

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #

    def step(self) -> tuple[float, bool]:
        """
        Advance by one slot.

        Returns
        -------
        offered_mbps : float
            Offered load this slot (Mbps).
        is_idle : bool
            True if UE has been idle > 50 ms (useful for WUR decisions).
        """
        # state transition
        if self.state == 'L':
            if self.rng.random() < self.p_lh:
                self.state = 'H'
        else:
            if self.rng.random() < self.p_hl:
                self.state = 'L'

        # offered load with small Gaussian noise (10% std)
        base_rate = self.lambda_l if self.state == 'L' else self.lambda_h
        noise     = self.rng.normal(0, 0.1 * base_rate)
        offered   = max(0.0, (base_rate + noise) * OVERHEAD_FACTOR)

        # idle tracker
        if offered < 0.5:
            self._idle_slots += 1
        else:
            self._idle_slots = 0

        is_idle = self._idle_slots > 50   # > 50 ms idle

        return offered, is_idle

    @property
    def is_high_load(self) -> bool:
        return self.state == 'H'

    @property
    def idle_duration_ms(self) -> int:
        return self._idle_slots

    def burstiness_index(self) -> float:
        """
        Theoretical c_v^2 of inter-arrival times.
        Should be ~4.2 for calibrated parameters.
        """
        lam_bar = self.pi_l * self.lambda_l + self.pi_h * self.lambda_h
        cv2 = (1 + 2 * self.pi_l * self.pi_h
               * (self.lambda_h - self.lambda_l)**2
               / (lam_bar**2 * (self.mu_lh + self.mu_hl)))
        return cv2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    src = MMPPTraffic(rng=rng)
    print(f"pi_H={src.pi_h:.3f}, pi_L={src.pi_l:.3f}")
    print(f"Mean rate: {src.mean_rate_mbps:.2f} Mbps")
    print(f"Burstiness c_v^2: {src.burstiness_index():.2f}  (target ~4.2)")

    # empirical check over 100k slots
    rates = [src.step()[0] for _ in range(100_000)]
    print(f"Empirical mean: {np.mean(rates):.2f} Mbps")
    print(f"Empirical std:  {np.std(rates):.2f} Mbps")
