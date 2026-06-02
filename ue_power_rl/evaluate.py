"""
Evaluation script.

Runs DQN agent + all 4 baselines over 100 episodes (500k slots each),
collects per-slot and per-episode metrics, saves to results/eval/.

Produces the data needed for all paper figures:
  Fig 1: Power vs traffic load
  Fig 2: EE reward convergence
  Fig 3: Latency CDF
  Fig 4: Ablation (joint vs single-parameter)

Usage:
    python -m ue_power_rl.evaluate --model results/models/v10kmh_d200m_s42/dqn_final
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from ue_power_rl.env      import UEPowerEnv
from ue_power_rl.baselines import (AlwaysOnPolicy, RandomPolicy,
                                    DRXPolicy, OraclePolicy)


# --------------------------------------------------------------------------- #
# Evaluation runner
# --------------------------------------------------------------------------- #
def evaluate_policy(policy, env_kwargs: dict,
                    n_episodes: int = 20,
                    seed_offset: int = 1000) -> pd.DataFrame:
    """
    Roll out a policy for n_episodes.

    Returns a DataFrame with one row per episode:
        ee_gbits_j, energy_mj, mean_lat_ms, mean_power_mw,
        pct_lat_violated, mean_tput_mbps, bwp_full_fraction
    """
    is_sb3 = hasattr(policy, "predict") and hasattr(policy, "policy")

    records = []
    for ep in range(n_episodes):
        env = UEPowerEnv(seed=seed_offset + ep, **env_kwargs)
        obs, _ = env.reset(seed=seed_offset + ep)

        ep_power     = []
        ep_lat       = []
        ep_tput      = []
        ep_offered   = []
        ep_bwp_full  = []
        ep_energy_mj = 0.0
        ep_bits      = 0.0

        done = False
        info_last = {}

        # DRX policy needs reset
        if hasattr(policy, "reset"):
            policy.reset()

        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            ep_power.append(info["power_mw"])
            ep_lat.append(info["latency_ms"])
            ep_tput.append(info["throughput_mbps"])
            ep_offered.append(info["offered_mbps"])
            ep_bwp_full.append(float(info["bwp_full"]))
            ep_energy_mj += info["power_mw"] * 1e-3       # mW*ms -> mJ
            ep_bits      += info["throughput_mbps"] * 1e6 * 1e-3  # bits
            info_last = info

        # EE = bits served per Joule (Gbits/J)
        ee = (ep_bits / 1e9) / (ep_energy_mj / 1e3) if ep_energy_mj > 0 else 0

        # Power saving vs always-on (975 mW baseline)
        mean_power    = float(np.mean(ep_power))
        power_saving_pct = (975.0 - mean_power) / 975.0 * 100

        # Throughput efficiency: served / offered (only during active traffic slots)
        ep_tput_arr    = np.array(ep_tput)
        ep_offered_arr = np.array(ep_offered)
        active_mask    = ep_offered_arr > 0.5
        if active_mask.sum() > 0:
            tput_efficiency = float(np.mean(
                np.minimum(ep_tput_arr[active_mask] / ep_offered_arr[active_mask], 1.0)
            ) * 100)
        else:
            tput_efficiency = 100.0

        records.append({
            "episode":           ep,
            "ee_gbits_j":        ee,
            "energy_mj":         ep_energy_mj,
            "mean_power_mw":     mean_power,
            "power_saving_pct":  power_saving_pct,
            "tput_efficiency_pct": tput_efficiency,
            "mean_lat_ms":       float(np.mean(ep_lat)),
            "p95_lat_ms":        float(np.percentile(ep_lat, 95)),
            "p99_lat_ms":        float(np.percentile(ep_lat, 99)),
            "pct_lat_violated":  float(np.mean(np.array(ep_lat) > 10.0) * 100),
            "mean_tput_mbps":    float(np.mean(ep_tput)),
            "mean_offered_mbps": float(np.mean(ep_offered)),
            "bwp_full_frac":     float(np.mean(ep_bwp_full)),
            "lat_samples":       ep_lat,
            "power_samples":     ep_power,
            "tput_samples":      ep_tput,
            "offered_samples":   list(ep_offered_arr),
        })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Collect results for all policies
# --------------------------------------------------------------------------- #
def run_all(model_path: str,
            velocity_kmh: float = 10.0,
            d2d_m: float = 200.0,
            n_episodes: int = 20,
            results_dir: str = "results"):

    env_kwargs = dict(velocity_kmh=velocity_kmh, d2d_m=d2d_m, ep_len_slots=5_000)
    os.makedirs(os.path.join(results_dir, "eval"), exist_ok=True)

    policies = {
        "DQN (ours)":   None,      # loaded below
        "3GPP DRX":     DRXPolicy(),
        "Always-on":    AlwaysOnPolicy(),
        "Random":       RandomPolicy(seed=99),
    }

    # load DQN
    try:
        model = DQN.load(model_path)
        policies["DQN (ours)"] = model
        print(f"DQN model loaded from {model_path}")
    except Exception as e:
        print(f"Could not load DQN model: {e}")
        print("Running baselines only.")
        del policies["DQN (ours)"]

    all_results = {}
    for name, policy in policies.items():
        print(f"  Evaluating: {name} ...")
        df = evaluate_policy(policy, env_kwargs,
                             n_episodes=n_episodes, seed_offset=2000)
        all_results[name] = df

        tag = f"v{int(velocity_kmh)}kmh"
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        df.drop(columns=["lat_samples","power_samples","tput_samples","offered_samples"]
                ).to_csv(os.path.join(results_dir, "eval",
                                       f"{safe_name}_{tag}.csv"), index=False)

        print(f"    EE:           {df.ee_gbits_j.mean():.3f} ± {df.ee_gbits_j.std():.3f} Gbits/J")
        print(f"    Power:        {df.mean_power_mw.mean():.1f} mW  (saving {df.power_saving_pct.mean():.1f}% vs always-on)")
        print(f"    Tput eff:     {df.tput_efficiency_pct.mean():.1f}% of offered load served")
        print(f"    Latency p95:  {df.p95_lat_ms.mean():.1f} ms")
        print(f"    Lat violated: {df.pct_lat_violated.mean():.1f}%")

    # summary table
    print("\n" + "="*90)
    print(f"{'Policy':<18} | {'EE (Gbits/J)':>12} | {'Power(mW)':>10} | "
          f"{'PwrSave%':>9} | {'TputEff%':>9} | {'Lat p95':>8} | {'Viol%':>6}")
    print("-"*90)
    for name, df in all_results.items():
        print(f"{name:<18} | {df.ee_gbits_j.mean():>8.3f}±{df.ee_gbits_j.std():.3f} | "
              f"{df.mean_power_mw.mean():>10.1f} | "
              f"{df.power_saving_pct.mean():>8.1f}% | "
              f"{df.tput_efficiency_pct.mean():>8.1f}% | "
              f"{df.p95_lat_ms.mean():>8.1f} | "
              f"{df.pct_lat_violated.mean():>5.1f}%")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,   default="results/models/v10kmh_d200m_s42/dqn_final")
    parser.add_argument("--velocity", type=float, default=10.0)
    parser.add_argument("--distance", type=float, default=200.0)
    parser.add_argument("--episodes", type=int,   default=20)
    parser.add_argument("--results",  type=str,   default="results")
    args = parser.parse_args()

    run_all(args.model, args.velocity, args.distance,
            args.episodes, args.results)
