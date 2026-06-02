"""
Baseline policies for comparison against the DQN agent.

1. AlwaysOnPolicy     — b=1, theta=1, SSB=5ms always
2. DRXPolicy          — 3GPP Rel-18 DRX state machine (Rel-18 spec)
3. RandomPolicy       — uniform random over action space
4. OraclePolicy       — value iteration on the discretised MDP (offline, needs VI table)

All policies implement .predict(obs, info) -> action (int, same as gym action).
"""

import numpy as np
from ..env.ue_env import ACTION_MAP, N_ACTIONS


def _find_action(bwp_full: bool, theta: int, ssb: int) -> int:
    return ACTION_MAP.index((bwp_full, theta, ssb))


# --------------------------------------------------------------------------- #
class AlwaysOnPolicy:
    """Full BWP, most sensitive WUR, shortest SSB. Upper QoS bound."""
    _action = _find_action(True, 1, 5)

    def predict(self, obs, info=None):
        return self._action, None


# --------------------------------------------------------------------------- #
class RandomPolicy:
    """Uniform random. Lower bound — establishes non-trivial problem."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, info=None):
        return int(self.rng.integers(N_ACTIONS)), None


# --------------------------------------------------------------------------- #
class DRXPolicy:
    """
    3GPP Rel-18 DRX power-saving heuristic.

    State machine:
        active      : recently had data (tau < T_inact)
        on_duration : periodic wake within DRX cycle
        sleep       : otherwise

    Parameters match 3GPP TS 38.321 DRX IE defaults for a UE in
    connected mode with power-saving extensions enabled.
    """

    T_INACT_MS   = 100   # inactivity timer (slots == ms here)
    T_DRX_MS     = 40    # long DRX cycle
    T_ON_MS      = 2     # on-duration within DRX cycle
    T_SHORT_MS   = 10    # short DRX cycle (used when transitioning to long)
    T_SHORT_CYCLE = 4    # slots in short DRX before switching to long

    def __init__(self):
        self._slot          = 0
        self._tau_slots     = 0
        self._drx_state     = 'active'
        self._short_count   = 0

    def predict(self, obs, info=None):
        """
        obs = [rho, q, cqi, bwp, tau_norm]
        tau_norm = tau_slots / 200
        """
        tau_slots = int(obs[4] * 200)  # de-normalise
        self._slot += 1

        # inactivity detection: use tau from state
        if tau_slots == 0:
            self._tau_slots = 0
            self._drx_state  = 'active'
            self._short_count = 0
        else:
            self._tau_slots = tau_slots

        if self._drx_state == 'active':
            if self._tau_slots < self.T_INACT_MS:
                # stay active
                action = _find_action(True, 1, 5)
            else:
                self._drx_state   = 'short_drx'
                self._short_count = 0
                action = _find_action(False, 2, 10)

        elif self._drx_state == 'short_drx':
            self._short_count += 1
            slot_in_cycle = self._slot % self.T_SHORT_MS
            if slot_in_cycle < self.T_ON_MS:
                action = _find_action(False, 2, 10)
            else:
                action = _find_action(False, 2, 20)
            if self._short_count >= self.T_SHORT_CYCLE * self.T_SHORT_MS:
                self._drx_state = 'long_drx'

        else:   # long_drx
            slot_in_cycle = self._slot % self.T_DRX_MS
            if slot_in_cycle < self.T_ON_MS:
                # on-duration: wake up, BWP-lite
                action = _find_action(False, 2, 20)
            else:
                # sleep: WUR only
                action = _find_action(False, 3, 40)

        return action, None

    def reset(self):
        self._slot        = 0
        self._tau_slots   = 0
        self._drx_state   = 'active'
        self._short_count = 0


# --------------------------------------------------------------------------- #
class OraclePolicy:
    """
    Value iteration oracle on the discretised 14,400-state MDP.
    Requires pre-computed Q-table. If not available, falls back to DRX.

    Used as a theoretical upper bound in the results section.
    """

    def __init__(self, q_table_path: str = None):
        self._q_table = None
        if q_table_path:
            try:
                self._q_table = np.load(q_table_path)
                print(f"Oracle Q-table loaded from {q_table_path}")
            except FileNotFoundError:
                print("Oracle Q-table not found, falling back to DRX.")
        self._fallback = DRXPolicy()

    def predict(self, obs, info=None):
        if self._q_table is None:
            return self._fallback.predict(obs, info)
        state_idx = self._obs_to_idx(obs)
        action = int(np.argmax(self._q_table[state_idx]))
        return action, None

    @staticmethod
    def _obs_to_idx(obs) -> int:
        """Map continuous obs to discrete state index (must match VI discretisation)."""
        rho_b  = int(np.clip(obs[0] * 10, 0, 9))
        q_b    = int(np.clip(obs[1] *  8, 0, 7))
        cqi_b  = int(np.clip(obs[2] * 15, 0, 14))
        bwp_b  = int(obs[3] > 0.5)
        tau_b  = int(np.clip(obs[4] *  6, 0, 5))
        return (rho_b * 8 * 15 * 2 * 6
                + q_b * 15 * 2 * 6
                + cqi_b * 2 * 6
                + bwp_b * 6
                + tau_b)
