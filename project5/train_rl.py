# train_rl.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .policy_net import PolicyNet
from .grid_env import GridEnv


# -----------------------------
# Hyperparameters & Toggles (defaults; can be overridden via CLI)
# -----------------------------
GAMMA = 0.99
LR = 3e-4
N_TRAJS_PER_UPDATE = 16
MAX_STEPS_PER_EPISODE = 40
NUM_UPDATES = 500

# Variance reduction options (pick one in practice)
USE_BASELINE = True          # moving-average scalar baseline
NORMALIZE_RETURNS = False    # batch-wise standardization (set True if not using baseline)

# Extra training stabilizers
ENTROPY_BETA = 1e-3
BASELINE_BETA = 0.9
moving_baseline = 0.0  # updated during training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# strict(er) reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Train REINFORCE on 5x5 grid")
    p.add_argument("--updates", type=int, default=NUM_UPDATES)
    p.add_argument("--n", type=int, default=N_TRAJS_PER_UPDATE, help="trajectories per update")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--entropy-beta", type=float, default=ENTROPY_BETA)
    p.add_argument("--use-baseline", action="store_true", default=USE_BASELINE)
    p.add_argument("--normalize-returns", action="store_true", default=NORMALIZE_RETURNS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--grid-json", type=str, default="", help="Path to 5x5 int grid JSON file")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    return p

# -----------------------------
# Utilities
# -----------------------------
def to_tensor_state(state_np: np.ndarray) -> torch.Tensor:
    """
    state_np: numpy array shaped (6, 5, 5), float32
    returns: tensor [1, 6, 5, 5] on DEVICE
    """
    return torch.from_numpy(state_np).float().unsqueeze(0).to(DEVICE)

def discount_rewards(rewards, gamma):
    """Compute discounted returns G_t for one episode."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def run_one_trajectory(env, policy, max_steps=MAX_STEPS_PER_EPISODE):
    """
    Roll out 1 episode using the current policy (training mode, gradients ON).
    Returns:
      log_probs:   list of log π(a_t|s_t)
      rewards:     list of r_t
      entropies:   list of H[π(·|s_t)]
    """
    log_probs, rewards, entropies = [], [], []

    state = env.reset()
    for _ in range(max_steps):
        state_t = to_tensor_state(state)
        # Gradients enabled during training rollout:
        action_probs = policy(state_t).squeeze(0)  # [4]
        dist = Categorical(action_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        next_state, reward, done = env.step(action.item())
        rewards.append(reward)

        state = next_state
        if done:
            break

    return log_probs, rewards, entropies

def reinforce_update(policy, optimizer, trajectories, gamma=GAMMA,
                     normalize_returns=NORMALIZE_RETURNS, use_baseline=USE_BASELINE):
    """
    trajectories: list of (log_probs, rewards, entropies) tuples
    """
    global moving_baseline
    all_returns, all_log_probs, all_entropies = [], [], []

    for log_probs, rewards, entropies in trajectories:
        returns = discount_rewards(rewards, gamma)
        all_returns.extend(returns)
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)

    returns = torch.tensor(all_returns, dtype=torch.float32, device=DEVICE)

    # Option 1: subtract a moving-average scalar baseline
    if use_baseline:
        moving_baseline = BASELINE_BETA * moving_baseline + (1 - BASELINE_BETA) * returns.mean().item()
        returns = returns - moving_baseline

    # Option 2: batch-wise normalization (zero-mean, unit-std)
    if normalize_returns:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs_tensor = torch.stack(all_log_probs)   # [sum(T_i)]
    entropy_tensor  = torch.stack(all_entropies)   # [sum(T_i)]

    # REINFORCE loss with entropy bonus (maximize entropy → subtract in loss)
    loss = -(log_probs_tensor * returns).sum() - ENTROPY_BETA * entropy_tensor.sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), returns.mean().item()

# -----------------------------
# Unit checks
# -----------------------------
def _assert_env_rewards():
    # 1) cheese (+10)
    g = np.zeros((5,5), dtype=np.int32); g[2,2]=1; g[2,3]=4
    e = GridEnv(g); e.reset(); _, r, done = e.step(3)  # right
    assert r == 10 and done, "Cheese should give +10 and end episode."

    # 2) trap (-50)
    g = np.zeros((5,5), dtype=np.int32); g[2,2]=1; g[2,3]=3
    e = GridEnv(g); e.reset(); _, r, done = e.step(3)
    assert r == -50 and done, "Trap should give -50 and end episode."

    # 3) empty / wall bump (-0.2)
    g = np.zeros((5,5), dtype=np.int32); g[0,0]=1  # top-left; move up bumps wall
    e = GridEnv(g); e.reset(); _, r, done = e.step(0)  # up
    assert abs(r + 0.2) < 1e-6 and not done, "Empty or wall bump should be -0.2."

def _assert_policy_output(policy):
    with torch.no_grad():
        dummy = np.zeros((6,5,5), dtype=np.float32); dummy[1,2,2]=1.0
        p = policy(to_tensor_state(dummy)).squeeze(0)
        assert torch.isfinite(p).all(), "Policy produced NaN/Inf."
        s = p.sum().item()
        assert 0.999 <= s <= 1.001, f"Policy probs must sum to 1, got {s}"

# -----------------------------
# Evaluation (greedy)
# -----------------------------
def evaluate(policy, env, episodes=100, max_steps=40):
    policy.eval()
    succ, totals, steps = 0, [], []
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            total, t = 0.0, 0
            while True:
                probs = policy(to_tensor_state(s)).squeeze(0)
                a = torch.argmax(probs).item()   # greedy eval
                s, r, done = env.step(a)
                total += r; t += 1
                if done or t >= max_steps:
                    break
            totals.append(total); steps.append(t)
            succ += (total >= 10.0)  # reaching any cheese yields +10 once
    print(f"Eval: mean_return={np.mean(totals):.2f}  success={succ/episodes:.1%}  mean_steps={np.mean(steps):.1f}")

def evaluate_random(env, episodes=50, max_steps=40):
    totals = []
    for _ in range(episodes):
        s = env.reset(); total=0; t=0
        while True:
            a = np.random.randint(0,4)
            s, r, done = env.step(a)
            total += r; t += 1
            if done or t>=max_steps: break
        totals.append(total)
    print(f"Random baseline: mean_return={np.mean(totals):.2f}")

# -----------------------------
# Main
# -----------------------------
def main():
    args = build_arg_parser().parse_args()
    global GAMMA, LR, ENTROPY_BETA, USE_BASELINE, NORMALIZE_RETURNS, SEED, MAX_STEPS_PER_EPISODE, N_TRAJS_PER_UPDATE, NUM_UPDATES
    GAMMA, LR = args.gamma, args.lr
    ENTROPY_BETA = args.entropy_beta
    USE_BASELINE = args.use_baseline
    NORMALIZE_RETURNS = args.normalize_returns
    SEED = args.seed
    MAX_STEPS_PER_EPISODE = args.max_steps
    N_TRAJS_PER_UPDATE = args.n
    NUM_UPDATES = args.updates

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"Device: {DEVICE}, Seed: {SEED}, USE_BASELINE={USE_BASELINE}, NORMALIZE_RETURNS={NORMALIZE_RETURNS}")

    # Load grid (optional JSON)
    if args.grid_json and os.path.exists(args.grid_json):
        import json
        with open(args.grid_json, "r") as f:
            grid = np.array(json.load(f), dtype=np.int32)
    else:
        # default example layout
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 0, 1, 0, 0],  # mouse at (2,2)
            [0, 4, 0, 5, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int32)

    env = GridEnv(grid)
    policy = PolicyNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # quick sanity tests
    _assert_env_rewards()
    _assert_policy_output(PolicyNet().to(DEVICE))

    best_avg_return = -1e9
    try:
        for update in range(1, NUM_UPDATES + 1):
            policy.train()
            trajectories = []
            total_return_per_traj = []

            # Collect N trajectories
            for _ in range(N_TRAJS_PER_UPDATE):
                log_probs, rewards, entropies = run_one_trajectory(env, policy)
                trajectories.append((log_probs, rewards, entropies))
                total_return_per_traj.append(sum(rewards))

            # Policy gradient step
            loss, mean_norm_return = reinforce_update(
                policy, optimizer, trajectories,
                normalize_returns=NORMALIZE_RETURNS,
                use_baseline=USE_BASELINE
            )

            # Logging & best checkpoint every 10 updates
            if update % 10 == 0:
                avg_return = float(np.mean(total_return_per_traj))
                print(f"[{update:04d}] loss={loss:.3f}  avg_return={avg_return:.2f}  mean_norm_return={mean_norm_return:.2f}")
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    tmp = "policy_net_best.tmp"
                    torch.save(policy.state_dict(), tmp)
                    os.replace(tmp, "policy_net_best.pt")  # atomic save
                    print(f"  ↳ new best avg_return={best_avg_return:.2f} (checkpoint saved)")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user. Saving current model...")

    # Final saves (atomic)
    tmp_final = "policy_net.tmp"
    torch.save(policy.state_dict(), tmp_final)
    os.replace(tmp_final, "policy_net.pt")
    print("Training done. Saved to policy_net.pt")

    # Baselines & evaluation
    evaluate_random(env, episodes=50, max_steps=MAX_STEPS_PER_EPISODE)
    evaluate(policy, env, episodes=args.eval_episodes, max_steps=MAX_STEPS_PER_EPISODE)

if __name__ == "__main__":
    main()
