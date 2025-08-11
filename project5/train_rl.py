# train_rl.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from policy_net import PolicyNet         # your PolicyNet class
from grid_env import GridEnv             # the env we wrote
# If these are in the same file/folder, make sure PYTHONPATH or relative imports are correct.

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 1e-3
N_TRAJS_PER_UPDATE = 16
MAX_STEPS_PER_EPISODE = 40
NUM_UPDATES = 500
NORMALIZE_RETURNS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Utilities
# -----------------------------
def to_tensor_state(state_np):
    """
    state_np: numpy array shaped (6, 5, 5), float32
    returns: tensor [1, 6, 5, 5]
    """
    return torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)

def discount_rewards(rewards, gamma):
    """ Compute discounted returns G_t for one episode. """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def run_one_trajectory(env, policy, max_steps=MAX_STEPS_PER_EPISODE):
    """
    Roll out 1 episode using the current policy.
    Returns:
      log_probs:   list of log π(a_t|s_t)
      rewards:     list of r_t
      states:      (optional) for debugging/visualization
    """
    log_probs = []
    rewards = []

    state = env.reset()
    for t in range(max_steps):
        state_t = to_tensor_state(state)
        with torch.no_grad():
            action_probs = policy(state_t).squeeze(0)     # [4]
        dist = Categorical(action_probs)
        action = dist.sample()

        # Store log prob for gradient later
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)

        # Step env
        next_state, reward, done = env.step(action.item())
        rewards.append(reward)

        state = next_state
        if done:
            break

    return log_probs, rewards

def reinforce_update(policy, optimizer, trajectories, gamma=GAMMA, normalize_returns=NORMALIZE_RETURNS):
    """
    trajectories: list of (log_probs, rewards) tuples
    """
    all_returns = []
    all_log_probs = []

    # Compute returns
    for log_probs, rewards in trajectories:
        returns = discount_rewards(rewards, gamma)
        all_returns.extend(returns)
        all_log_probs.extend(log_probs)

    returns = torch.tensor(all_returns, dtype=torch.float32, device=DEVICE)

    if normalize_returns:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs_tensor = torch.stack(all_log_probs)  # shape [sum(T_i)]
    loss = -(log_probs_tensor * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), returns.mean().item()

# -----------------------------
# Main
# -----------------------------
def main():
    # Example fixed grid (replace with what you export from your UI, or randomize)
    # 0 empty, 1 mouse, 2 wall, 3 trap, 4 cheese, 5 organic
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

    for update in range(1, NUM_UPDATES + 1):
        trajectories = []
        total_return_per_traj = []

        # Collect N trajectories
        for _ in range(N_TRAJS_PER_UPDATE):
            log_probs, rewards = run_one_trajectory(env, policy)
            trajectories.append((log_probs, rewards))
            total_return_per_traj.append(sum(rewards))

        # Policy gradient step
        loss, mean_norm_return = reinforce_update(policy, optimizer, trajectories)

        if update % 10 == 0:
            avg_return = np.mean(total_return_per_traj)
            print(f"[{update:04d}] loss={loss:.3f}  avg_return={avg_return:.2f}  mean_norm_return={mean_norm_return:.2f}")

    # (Optional) Save model
    torch.save(policy.state_dict(), "policy_net.pt")
    print("Training done. Saved to policy_net.pt")

if __name__ == "__main__":
    main()
