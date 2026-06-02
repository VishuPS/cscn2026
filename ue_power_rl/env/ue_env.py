"""
UE Power Saving Gymnasium Environment.

Implements the MDP defined in Section 2 of the paper:
  S = (rho, q, CQI, b, tau)
  A = (bwp_mode, wur_threshold, ssb_period)  — 24 discrete joint actions
  R = EE - latency_penalty - tput_penalty - switch_cost - access_penalty

Compatible with Stable-Baselines3 DQN.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .channel    import UMaChannel
from .traffic    import MMPPTraffic
from .power_model import UEPowerModel, UEHardwareProfile
from .wur        import WURDetector, pmd_fast as pmd


# --------------------------------------------------------------------------- #
# Action space definition
# --------------------------------------------------------------------------- #
BWP_MODES    = [False, True]        # False=BWP-lite, True=full-BW
WUR_THETAS   = [1, 2, 3]           # detection threshold levels
SSB_PERIODS  = [5, 10, 20, 40]     # ms

# Build flat action index -> (bwp_mode, theta, ssb_period_ms)
ACTION_MAP = []
for bwp in BWP_MODES:
    for theta in WUR_THETAS:
        for ssb in SSB_PERIODS:
            ACTION_MAP.append((bwp, theta, ssb))
N_ACTIONS = len(ACTION_MAP)   # = 2 * 3 * 4 = 24


# --------------------------------------------------------------------------- #
# State space normalisation bounds
# --------------------------------------------------------------------------- #
# [rho, q_norm, cqi_norm, bwp, tau_norm]
OBS_LOW  = np.array([0.0, 0.0,  0.0, 0.0, 0.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0,  1.0, 1.0, 1.0], dtype=np.float32)


class UEPowerEnv(gym.Env):
    """
    Single-UE 6G NR power-saving environment.

    Parameters
    ----------
    velocity_kmh : UE velocity (3, 10, or 60 km/h)
    d2d_m        : UE-gNB distance (m)
    ep_len_slots : episode length in 1 ms slots
    reward_weights : dict with keys lambda1, lambda2, mu_s, mu_ia
    seed         : RNG seed
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 velocity_kmh:    float = 10.0,
                 d2d_m:           float = 200.0,
                 ep_len_slots:    int   = 5_000,
                 reward_weights:  dict  = None,
                 seed:            int   = None):

        super().__init__()
        self.velocity_kmh  = velocity_kmh
        self.d2d_m         = d2d_m
        self.ep_len_slots  = ep_len_slots
        self.seed_val      = seed

        # reward weights — all penalties now 0..1 range
        rw = reward_weights or {}
        self.lam1   = rw.get("lambda1", 0.5)   # latency penalty weight
        self.lam2   = rw.get("lambda2", 0.3)   # throughput penalty weight
        self.mu_s   = rw.get("mu_s",    0.05)  # switching cost (small)
        self.mu_ia  = rw.get("mu_ia",   0.1)   # access time penalty

        # QoS targets
        self.D_max_ms   = 10.0    # max latency (6G URLLC target)
        self.R_min_mbps = 1.0     # min throughput
        self.rho_th     = 0.3     # load threshold for access penalty

        # spaces
        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH,
                                             dtype=np.float32)

        # will be initialised in reset()
        self.rng       = None
        self.channel   = None
        self.traffic   = None
        self.power_mdl = UEPowerModel()
        self.wur       = None

        # episode state
        self._slot       = 0
        self._bwp_full   = True
        self._wur_theta  = 1
        self._ssb_period = 5
        self._buffer_mb  = 0.0
        self._tau_slots  = 0    # idle timer
        self._rho_ewma   = 0.0
        self._prev_bwp   = True

        # metrics accumulators
        self._ep_energy_mj   = 0.0
        self._ep_bits        = 0.0
        self._ep_latency_sum = 0.0
        self._ep_steps       = 0

    # ----------------------------------------------------------------------- #
    # Gymnasium API
    # ----------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else self.seed_val
        self.rng = np.random.default_rng(rng_seed)

        self.channel = UMaChannel(
            velocity_kmh=self.velocity_kmh,
            d2d_m=self.d2d_m,
            rng=np.random.default_rng(self.rng.integers(1e9))
        )
        self.traffic = MMPPTraffic(
            rng=np.random.default_rng(self.rng.integers(1e9))
        )
        self.wur = WURDetector(
            rng=np.random.default_rng(self.rng.integers(1e9))
        )

        self._slot       = 0
        self._bwp_full   = True
        self._wur_theta  = 1
        self._ssb_period = 5
        self._buffer_mb  = 0.0
        self._tau_slots  = 0
        self._rho_ewma   = 0.0
        self._prev_bwp   = True

        self._ep_energy_mj   = 0.0
        self._ep_bits        = 0.0
        self._ep_latency_sum = 0.0
        self._ep_steps       = 0

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        # decode action
        new_bwp, new_theta, new_ssb = ACTION_MAP[action]
        self._prev_bwp   = self._bwp_full
        self._bwp_full   = new_bwp
        self._wur_theta  = new_theta
        self._ssb_period = new_ssb

        # 1. Channel step
        cqi = self.channel.step()
        snr_db = self.channel.current_snr_db()

        # 2. Traffic step
        offered_mbps, is_idle = self.traffic.step()

        # 3. WUR decision: gNB sends WUS if UE is in BWP-lite and has downlink data
        # Fixed: WUR can trigger regardless of tau — any time UE is not full-BW
        ue_lite  = not self._bwp_full
        wus_sent = (offered_mbps > 0.5) and ue_lite
        wake_up, _ = self.wur.decide(self._wur_theta, snr_db, wus_sent)

        # 4. Determine radio activity this slot
        if self._bwp_full:
            n_sym_active = 14                      # full active
        elif ue_lite and wus_sent and wake_up:
            n_sym_active = 7                       # WUS detected: partial wake
        elif ue_lite and wus_sent and not wake_up:
            n_sym_active = 0                       # missed WUS: stay asleep
        elif ue_lite and not wus_sent:
            n_sym_active = 3                       # BWP-lite monitoring only
        else:
            n_sym_active = 7

        # 5. Achievable throughput
        if n_sym_active > 0:
            r_mbps = (self.channel.throughput_mbps(bwp_full=self._bwp_full)
                      * (n_sym_active / 14.0))
        else:
            r_mbps = 0.0

        # 6. Buffer dynamics (1 ms slot, 1e-3 s)
        slot_s = 1e-3
        arrived_mb  = offered_mbps * slot_s        # MB arrived this slot
        served_mb   = min(r_mbps * slot_s, self._buffer_mb + arrived_mb)
        self._buffer_mb = max(0.0, self._buffer_mb + arrived_mb - served_mb)
        self._buffer_mb = min(self._buffer_mb, 1.0)   # cap at 1 MB

        # 7. Latency estimate (simple queuing: buffer / service_rate)
        if r_mbps > 0:
            d_ms = (self._buffer_mb / r_mbps) * 1e3   # ms
        else:
            d_ms = self.D_max_ms * 5 if self._buffer_mb > 0 else 0.0
        d_ms = float(np.clip(d_ms, 0, 200))

        # 8. Power consumption
        power_out = self.power_mdl.compute(
            bwp_full=self._bwp_full,
            wur_active=(not self._bwp_full),
            n_sym_active=n_sym_active,
            prev_bwp_full=self._prev_bwp,
            slot_ms=1.0
        )
        p_total_mw = power_out["total"] + self.power_mdl.ssb_overhead_mw(self._ssb_period)

        # 9. Update state variables
        alpha_ew = 0.1
        self._rho_ewma = ((1 - alpha_ew) * self._rho_ewma
                          + alpha_ew * min(offered_mbps / 20.0, 1.0))

        if offered_mbps < 0.5:
            self._tau_slots += 1
        else:
            self._tau_slots = 0

        self._slot += 1

        # 10. Reward computation
        # Design principle: reward must create a clear gradient toward the target
        # behaviour: full-BW when traffic active, BWP-lite/sleep when idle.
        # All terms normalised to [-1, +1] range.

        slot_s       = 1e-3
        p_j_per_slot = p_total_mw * 1e-3 * slot_s
        bits_served  = served_mb * 8e6

        # --- Component 1: Power saving reward (positive, 0..1) ---
        # Reward proportional to how much power saved vs always-on baseline
        p_always_on_mw = 975.0
        power_saving = np.clip((p_always_on_mw - p_total_mw) / p_always_on_mw,
                               0.0, 1.0)

        # --- Component 2: Throughput reward (positive, 0..1) ---
        # Only meaningful when traffic is present
        if offered_mbps > 0.5:
            tput_reward = np.clip(r_mbps / offered_mbps, 0.0, 1.0)
        else:
            tput_reward = 1.0   # no traffic = no throughput requirement

        # --- Component 3: Buffer penalty (negative, 0..1) ---
        # Directly penalise buffer build-up — this is what drives wake-up behaviour
        # Buffer > 0 when sleeping during traffic = immediate strong penalty
        buffer_penalty = np.clip(self._buffer_mb / 0.1, 0.0, 1.0)  # saturates at 100KB

        # --- Component 4: Latency penalty (negative, 0..1) ---
        lat_penalty = np.clip(d_ms / (self.D_max_ms * 2), 0.0, 1.0)

        # --- Component 5: Switching cost ---
        switch_penalty = self.mu_s * float(new_bwp != self._prev_bwp)

        # --- Weighted combination ---
        # Weights tuned so that:
        #   - sleeping during idle (no traffic): reward ~ 0.7 (power saving)
        #   - active during traffic, low latency: reward ~ 0.6
        #   - sleeping during traffic (buffer builds): reward ~ 0.7 - 0.9 = -0.2
        #   This forces the agent to wake up during traffic bursts
        w_power  = 0.35
        w_tput   = 0.35
        w_buffer = 0.55   # strong: buffer build-up must be punished hard
        w_lat    = 0.20

        reward = (w_power  * power_saving
                + w_tput   * tput_reward
                - w_buffer * buffer_penalty
                - w_lat    * lat_penalty
                - switch_penalty)

        # 11. Accumulate episode metrics
        self._ep_energy_mj   += p_total_mw * slot_s   # mW * s = mJ
        self._ep_bits        += bits_served
        self._ep_latency_sum += d_ms
        self._ep_steps       += 1

        terminated = False
        truncated  = (self._slot >= self.ep_len_slots)

        info = {
            "power_mw":        p_total_mw,
            "throughput_mbps": r_mbps,
            "latency_ms":      d_ms,
            "cqi":             cqi,
            "bwp_full":        self._bwp_full,
            "wur_theta":       self._wur_theta,
            "ssb_period":      self._ssb_period,
            "offered_mbps":    offered_mbps,
            "ee_gbits_j":      (bits_served/1e9) / (p_j_per_slot + 1e-12),
            "power_saving":    float(power_saving),
            "buffer_mb":       self._buffer_mb,
        }

        if truncated:
            ep_ee = (self._ep_bits / 1e9) / (self._ep_energy_mj / 1e3) if self._ep_energy_mj > 0 else 0
            info["episode_ee_gbits_j"]    = ep_ee
            info["episode_energy_mj"]     = self._ep_energy_mj
            info["episode_mean_lat_ms"]   = self._ep_latency_sum / max(self._ep_steps, 1)

        return self._get_obs(), reward, terminated, truncated, info

    # ----------------------------------------------------------------------- #
    # Internal
    # ----------------------------------------------------------------------- #

    def _get_obs(self) -> np.ndarray:
        """Return normalised observation vector [0,1]^5."""
        rho_norm  = float(np.clip(self._rho_ewma, 0, 1))
        q_norm    = float(np.clip(self._buffer_mb, 0, 1))
        cqi_norm  = float((self.channel.cqi - 1) / 14.0) if self.channel else 0.5
        bwp_norm  = float(self._bwp_full)
        tau_norm  = float(np.clip(self._tau_slots / 200.0, 0, 1))

        return np.array([rho_norm, q_norm, cqi_norm, bwp_norm, tau_norm],
                         dtype=np.float32)
