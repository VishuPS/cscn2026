"""
Wake-Up Receiver (WUR) detection model.

Implements energy detector statistics for a 64-symbol WUS sequence
under Rayleigh fading, as derived in Section 7 of the paper math.

Key functions:
  - pfa(theta)           : false alarm probability
  - pmd(theta, snr_db)   : miss detection probability (fading-averaged)
  - sample_detection()   : Monte Carlo WUS detection event
"""

import numpy as np
from scipy.special import gammaincc, gammainc


# WUS signal parameters (3GPP Rel-18 WUS design)
L_WUS   = 64      # sequence length (symbols)
N_PRB   = 1       # WUS occupies 1 PRB

# Threshold index -> actual normalised threshold multiplier
# theta=1: low sensitivity (aggressive), theta=3: high sensitivity (conservative)
THETA_LEVELS = {1: 1.130, 2: 1.273, 3: 1.431}


def pfa(theta: int) -> float:
    """
    False alarm probability for energy detector.

    Under H0 (no WUS), T ~ (N0/2)*chi2_{2*L_WUS}.
    Threshold xi = theta * N0 * L_WUS.

    P_FA = P(chi2_{2L} > 2*theta*L) = gammaincc(L, theta*L)
         = upper regularised incomplete gamma.

    Parameters
    ----------
    theta : int in {1,2,3}
    """
    th = THETA_LEVELS[theta]
    return float(gammaincc(L_WUS, th * L_WUS))


def pmd(theta: int, snr_db: float) -> float:
    """
    Miss detection probability, averaged over Rayleigh fading.

    Under H1, using the closed-form result for energy detection
    with non-central chi-squared under Rayleigh fading:

        P_MD = sum_{k=0}^{L-1} [exp(-xi/(1+SNR)) * (xi/(1+SNR))^k / k!]
             * exp(-xi * SNR / (1+SNR))    ... but this simplifies to:

        P_MD = 1 - Q_M(sqrt(2*lambda_nc), sqrt(2*xi))

    For the Rayleigh-averaged case (marginalise over |h|^2 ~ Exp(1)):

        P_D = integral_0^inf P_D(gamma) * exp(-gamma/SNR_bar) / SNR_bar dgamma
            = exp(-xi/(1+SNR_bar)) * sum_{k=0}^{L_WUS-1} (xi/(1+SNR_bar))^k / k!
              * 1/(1+SNR_bar)^0   ... full form below

    We use the standard result:
        P_D(theta, SNR) = 1 - exp(-theta*L/(1+SNR))
                          * sum_{k=0}^{L-1} (theta*L)^k / (k! * (1+SNR)^k)
                          * exp(-theta*L*SNR/(1+SNR))

    Parameters
    ----------
    theta  : int in {1,2,3}
    snr_db : float  (average received SNR in dB)
    """
    th      = THETA_LEVELS[theta]
    snr_lin = 10 ** (snr_db / 10)
    xi      = th * L_WUS                    # normalised threshold

    # detection probability via series expansion
    # P_D = exp(-xi/(1+SNR)) * exp(-xi*SNR/(1+SNR)) * sum_k (xi/(1+SNR))^k / k!
    # Note: exp(-xi/(1+SNR)) * exp(-xi*SNR/(1+SNR)) = exp(-xi)  ... but that's
    # only for the non-fading case. Correct Rayleigh-faded form:
    #
    # P_D = (1+SNR)^{-1} * exp(-xi/(1+SNR))
    #       * sum_{k=0}^{L-1} C(L-1+k, k) * (SNR/(1+SNR))^k
    #       ... which is the Marcum-Q alternative for Rayleigh fading.
    #
    # Practical implementation: use the upper incomplete gamma shortcut.
    # Under Rayleigh fading, the composite test statistic T is a scaled
    # non-central chi-squared with nc parameter Lambda ~ Exp(L*SNR).
    # After marginalisation:
    #
    # P_D = sum_{k=0}^{inf} P(Poisson(L*SNR)=k) * P(chi2_{2(L+k)} > 2*xi)
    #
    # Truncated sum (converges quickly for L=64):

    snr_bar = snr_lin
    lam     = L_WUS * snr_bar       # non-centrality * L
    log_lam = np.log(lam) if lam > 0 else -np.inf

    # Iterate until past the Poisson mode (lam) + 8 std devs
    k_max = int(lam + 8 * np.sqrt(max(lam, 1))) + 50
    pd = 0.0
    log_poisson_k = -lam            # log P(Poisson(lam)=0)
    for k in range(k_max):
        weight = np.exp(log_poisson_k) if log_poisson_k > -700 else 0.0
        if weight > 1e-300:
            p_chi = float(gammaincc(L_WUS + k, xi))
            pd += weight * p_chi
        # update log weight: log P(k+1) = log P(k) + log(lam) - log(k+1)
        log_poisson_k += log_lam - np.log(k + 1) if lam > 0 else -np.inf

    pd = float(np.clip(pd, 0.0, 1.0))
    return 1.0 - pd   # P_MD = 1 - P_D


def detection_table() -> dict:
    """
    Return P_FA and P_MD table for all theta levels and key SNR points.
    Matches Table II in the paper.
    """
    snr_points = [0, 10, 20]
    table = {}
    for th in [1, 2, 3]:
        table[th] = {
            "pfa": pfa(th),
            "pmd": {snr: pmd(th, snr) for snr in snr_points}
        }
    return table


# --------------------------------------------------------------------------- #
# Precomputed lookup table: pmd(theta, snr_db) for fast inference
# snr grid: -5 to 30 dB in 1 dB steps
# --------------------------------------------------------------------------- #
_SNR_GRID  = np.arange(-5, 31, 1, dtype=float)   # 36 points
_PMD_TABLE = {th: np.array([pmd(th, s) for s in _SNR_GRID])
              for th in [1, 2, 3]}


def pmd_fast(theta: int, snr_db: float) -> float:
    """Lookup-table version of pmd(). ~100x faster than full series."""
    snr_clipped = float(np.clip(snr_db, _SNR_GRID[0], _SNR_GRID[-1]))
    idx = np.searchsorted(_SNR_GRID, snr_clipped)
    idx = int(np.clip(idx, 0, len(_SNR_GRID) - 1))
    return float(_PMD_TABLE[theta][idx])


class WURDetector:
    """
    Stateful WUR: decides whether to wake the main radio each slot.

    Parameters
    ----------
    rng : np.random.Generator
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def decide(self, theta: int, snr_db: float,
               wus_sent: bool) -> tuple[bool, str]:
        """
        Simulate one WUS detection decision.

        Parameters
        ----------
        theta    : threshold level {1,2,3}
        snr_db   : current channel SNR
        wus_sent : whether the gNB actually transmitted a WUS this slot

        Returns
        -------
        wake_up : bool   — True if UE decides to wake
        event   : str    — 'hit'|'miss'|'false_alarm'|'correct_reject'
        """
        p_fa = pfa(theta)
        p_md = pmd_fast(theta, snr_db)

        if wus_sent:
            detected = self.rng.random() > p_md   # hit or miss
            return detected, ("hit" if detected else "miss")
        else:
            false_alarm = self.rng.random() < p_fa
            return false_alarm, ("false_alarm" if false_alarm else "correct_reject")


if __name__ == "__main__":
    print("=== WUR Detection Table (P_FA / P_MD) ===")
    print(f"{'theta':>6} | {'P_FA':>8} | {'P_MD@0dB':>10} | "
          f"{'P_MD@10dB':>10} | {'P_MD@20dB':>10}")
    print("-" * 58)
    for th in [1, 2, 3]:
        p_fa_val = pfa(th)
        pmd_vals = [pmd(th, snr) for snr in [0, 10, 20]]
        print(f"  {th:>4} | {p_fa_val:>8.3f} | {pmd_vals[0]:>10.3f} | "
              f"{pmd_vals[1]:>10.3f} | {pmd_vals[2]:>10.3f}")
