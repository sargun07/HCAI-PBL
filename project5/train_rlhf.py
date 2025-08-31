# train_rlhf.py
import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

from .policy_net import PolicyNet
from .grid_env import GridEnv
from .reward_model import RewardNet  # from Task 2

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 3e-4
N_TRAJS_PER_UPDATE = 16
MAX_STEPS_PER_EPISODE = 40
NUM_UPDATES = 400
NORMALIZE_RETURNS = True
ENTROPY_BETA = 1e-3
BASELINE_BETA = 0.9
KL_BETA = 0.05            # strength of KL penalty to the base policy
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

moving_baseline = 0.0

# -----------------------------
# Utilities
# -----------------------------
def onehot_from_codes(grid_codes):
    # grid_codes: (5,5) ints in [0..5]
    oh = np.zeros((6,5,5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            oh[grid_codes[i, j], i, j] = 1.0
    return oh

def to_tensor_state(state_np):
    # state_np: (6,5,5) float32
    return torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)

def to_tensor_batch(states_np_list):
    # list of (6,5,5) -> [T,6,5,5]
    x = np.stack(states_np_list, axis=0)
    return torch.from_numpy(x).float().to(DEVICE)

def discount_rewards(rews, gamma):
    G, out = 0.0, []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out

# -----------------------------
# Rollout using CURRENT policy (stochastic)
# Store states & actions; we ignore env's hand-coded reward here (Task 3 uses learned reward)
# -----------------------------
def run_one_trajectory(env, policy, max_steps=MAX_STEPS_PER_EPISODE):
    states_np, actions, entropies = [], [], []
    grid = env.reset()
    # env.reset() returns onehot state (6,5,5); make sure
    state_np = grid  # already (6,5,5)
    for _ in range(max_steps):
        st = to_tensor_state(state_np)            # [1,6,5,5]
        probs = policy(st).squeeze(0)             # [4]
        dist = Categorical(probs)
        a = dist.sample()

        actions.append(int(a.item()))
        entropies.append(dist.entropy())          # keep for entropy bonus
        states_np.append(state_np)                # keep numpy for reward_net later

        # Step env (uses true env dynamics; reward ignored for training signal)
        next_state_np, _, done = env.step(int(a.item()))
        state_np = next_state_np
        if done:
            # add final observed state (helps reward model see terminal)
            states_np.append(state_np)
            break

    return states_np, actions, entropies

# -----------------------------
# Compute REINFORCE loss with learned rewards and KL penalty
# -----------------------------
def reinforce_update(policy, optimizer, ref_policy, reward_net, trajectories):
    global moving_baseline

    # Flatten per-trajectory pieces
    all_states_np, all_actions, all_entropies = [], [], []
    traj_returns = []  # for logging

    # 1) Compute learned rewards for each trajectory
    for states_np, actions, entropies in trajectories:
        # Reward model scores per state (no discount yet)
        with torch.no_grad():
            S = to_tensor_batch(states_np)                 # [T,6,5,5]
            r_hat = reward_net(S).squeeze(-1)              # [T]
        # If we appended terminal at the end, it contributes too; fine.
        returns = discount_rewards(r_hat.cpu().numpy().tolist(), GAMMA)
        traj_returns.append(sum(returns))
        # Store to flat buffers
        all_states_np.extend(states_np)
        all_actions.extend(actions)
        # Clip entropies list length to actions length (entropies collected per action step)
        all_entropies.extend(entropies[:len(actions)])

    # 2) Assemble tensors
    states = to_tensor_batch(all_states_np)                         # [N,6,5,5]
    actions_t = torch.tensor(all_actions, dtype=torch.long, device=DEVICE)  # [N']
    # returns only for action-steps, align shapes
    # Re-compute learned rewards and returns aligned to actions (ignore the possible final terminal extra state)
    with torch.no_grad():
        r_hat_all = reward_net(states).squeeze(-1)                  # [N_total_states]
    # Build per-step returns aligned to action count
    # We need returns for each (s_t, a_t). Recompute per-trajectory to align lengths:
    per_step_returns = []
    idx = 0
    for states_np, actions, _ in trajectories:
        T = len(actions)                         # number of action steps in this traj
        # rewards for these T steps are the first T r_hat entries of this traj
        r_local = r_hat_all[idx: idx + T]        # [T]
        idx += len(states_np)                     # advance by total states in traj (including terminal)
        # compute discounted returns on CPU tensor -> back to DEVICE
        Gs = discount_rewards(r_local.cpu().numpy().tolist(), GAMMA)
        per_step_returns.extend(Gs)
    returns_t = torch.tensor(per_step_returns, dtype=torch.float32, device=DEVICE)  # [N']

    # 3) Baseline + normalization
    if True:  # baseline on
        moving_baseline = BASELINE_BETA * moving_baseline + (1 - BASELINE_BETA) * returns_t.mean().item()
        returns_t = returns_t - moving_baseline
    if NORMALIZE_RETURNS:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    # 4) Log-probs for TAKEN actions under current policy (same params as rollout)
    policy.train()
    logits = policy(states)                      # [N_total_states, 4]
    # Only keep rows that correspond to action steps. We computed logits for every stored state,
    # but actions exist only for the first part of each trajectory state list.
    # Simplest: rebuild indices of action-bearing rows.
    idxs = []
    count = 0
    for states_np, actions, _ in trajectories:
        idxs.extend(list(range(count, count + len(actions))))
        count += len(states_np)
    idxs_t = torch.tensor(idxs, dtype=torch.long, device=DEVICE)
    logits_act = logits.index_select(0, idxs_t)  # [N',4]
    dist_curr = Categorical(probs=logits_act)

    log_probs = dist_curr.log_prob(actions_t)    # [N']
    entropies = torch.stack(all_entropies).to(DEVICE)  # [N'] (already per action step)

    # 5) KL penalty to REFERENCE policy
    with torch.no_grad():
        ref_logits = ref_policy(states).index_select(0, idxs_t)  # [N',4]
        ref_probs = torch.clamp(ref_logits, 1e-8, 1-1e-8)

    curr_probs = torch.clamp(logits_act, 1e-8, 1-1e-8)
    kl = (curr_probs * (curr_probs.log() - ref_probs.log())).sum(dim=-1)  # [N']
    kl_term = kl.mean()

    # 6) Total loss: REINFORCE (maximize) + entropy bonus + KL penalty
    loss = -(log_probs * returns_t).mean() - ENTROPY_BETA * entropies.mean() + KL_BETA * kl_term

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return float(loss.item()), float(torch.tensor(traj_returns).mean().item()), float(kl_term.item())

# -----------------------------
# Main
# -----------------------------
def main(steps=50, gamma=0.99, entropy=1e-3, kl_coef=0.05, max_steps=40,
         trajs_per_update=16, lr=3e-4):
    """
    Fine-tune the policy with a learned reward (BT) + KL penalty to the base policy.

    Returns (ok: bool, info: dict) where info includes:
      - "ckpt": best checkpoint filename
      - "learned_return_mean": last update's avg learned return
      - "kl_to_ref": last update's mean KL to the reference policy
    """
    global GAMMA, ENTROPY_BETA, KL_BETA, MAX_STEPS_PER_EPISODE, N_TRAJS_PER_UPDATE, LR
    GAMMA = float(gamma)
    ENTROPY_BETA = float(entropy)
    KL_BETA = float(kl_coef)
    MAX_STEPS_PER_EPISODE = int(max_steps)
    N_TRAJS_PER_UPDATE = int(trajs_per_update)
    LR = float(lr)

    print(f"[RLHF] Device={DEVICE} | steps={steps} | gamma={GAMMA} | "
          f"entropy={ENTROPY_BETA} | kl={KL_BETA} | max_steps={MAX_STEPS_PER_EPISODE} | "
          f"n_trajs={N_TRAJS_PER_UPDATE} | lr={LR}")

    # 1) Environment (example fixed grid; your GridEnv can also randomize each update)
    grid_codes = np.array([
        [0,0,0,0,0],
        [0,2,0,3,0],
        [0,0,1,0,0],
        [0,4,0,5,0],
        [0,0,0,0,0],
    ], dtype=np.int32)
    env = GridEnv(grid_codes)

    # 2) Load base (reference) policy π_ref from your Task-1 checkpoint
    ref = PolicyNet().to(DEVICE)
    candidates = [
        os.path.join(os.getcwd(), "policy_net_best.pt"),
        os.path.join(os.getcwd(), "policy_net.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net_best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net.pt"),
    ]
    base_ckpt = next((p for p in candidates if os.path.exists(p)), None)
    if base_ckpt is None:
        raise FileNotFoundError("No trained model found in:\n" + "\n".join(candidates))
    ref.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))
    ref.eval()  # frozen

    # 3) Policy to fine-tune starts from the same base
    policy = PolicyNet().to(DEVICE)
    policy.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))

    # 4) Load learned reward model from Task 2 (Bradley–Terry)
    reward_net = RewardNet().to(DEVICE)
    if not os.path.exists("reward_net.pt"):
        raise FileNotFoundError("reward_net.pt not found. Train reward (BT) first.")
    reward_net.load_state_dict(torch.load("reward_net.pt", map_location=DEVICE))
    reward_net.eval()  # fixed during RLHF

    # 5) Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # 6) Train loop
    best_avg = -1e9
    last_kl, last_ret = None, None
    try:
        for update in range(1, int(steps) + 1):
            # Collect fresh on-policy trajectories
            trajectories = [run_one_trajectory(env, policy) for _ in range(N_TRAJS_PER_UPDATE)]
            # One REINFORCE update using learned reward + KL penalty
            loss, avg_return_learned, kl_val = reinforce_update(
                policy, optimizer, ref, reward_net, trajectories
            )
            last_kl, last_ret = float(kl_val), float(avg_return_learned)

            # Save "best" by learned return every 10 steps (cheap heuristic)
            if update % 10 == 0:
                print(f"[RLHF {update:03d}] loss={loss:.3f}  learned_return={avg_return_learned:.2f}  KL={kl_val:.4f}")
                if avg_return_learned > best_avg:
                    best_avg = avg_return_learned
                    tmp = "policy_net_rlhf_best.tmp"
                    torch.save(policy.state_dict(), tmp)
                    os.replace(tmp, "policy_net_rlhf_best.pt")
                    print("  ↳ new best; saved policy_net_rlhf_best.pt")
    finally:
        # Always save the final model as well
        tmp_final = "policy_net_rlhf.tmp"
        torch.save(policy.state_dict(), tmp_final)
        os.replace(tmp_final, "policy_net_rlhf.pt")
        print("Saved final RLHF model to policy_net_rlhf.pt")

    info = {
        "ckpt": "policy_net_rlhf_best.pt" if os.path.exists("policy_net_rlhf_best.pt") else "policy_net_rlhf.pt",
        "learned_return_mean": last_ret,
        "kl_to_ref": last_kl,
    }
    return True, info


if __name__ == "__main__":
    main()
