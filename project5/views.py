import json, os, torch, traceback, io, zipfile
import numpy as np
from django.http import FileResponse, JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
import time, uuid, json

from .policy_net import PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_POLICY = {} # lazy-loaded singleton


PREF_LOG = "preferences.jsonl"

# --- your existing in-memory layout store (keep yours if different) ---
CURRENT_LAYOUT = None

def index(request):
    return render(request, "project5/interface.html")

def _load_policy(kind="trained"):
    """
    kind: "trained" (policy_net*.pt) or "rlhf" (policy_net_rlhf*.pt)
    """
    if kind in _POLICY: 
        return _POLICY[kind]

    ckpt_candidates = {
        "trained": [
            os.path.join(os.getcwd(), "policy_net_best.pt"),
            os.path.join(os.getcwd(), "policy_net.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net_best.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net.pt"),
        ],
        "rlhf": [
            os.path.join(os.getcwd(), "policy_net_rlhf_best.pt"),
            os.path.join(os.getcwd(), "policy_net_rlhf.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net_rlhf_best.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_net_rlhf.pt"),
        ],
    }[kind]

    path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"No '{kind}' model found. Expected one of:\n" + "\n".join(ckpt_candidates))

    print(f"✅ Loading {kind} policy from: {path}")
    net = PolicyNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    _POLICY[kind] = net
    return net


def _one_hot_from_grid(grid_np):
    # grid_np: (5,5) ints 0..5 -> (6,5,5) float32
    state = np.zeros((6, grid_np.shape[0], grid_np.shape[1]), dtype=np.float32)
    for i in range(grid_np.shape[0]):
        for j in range(grid_np.shape[1]):
            state[grid_np[i, j], i, j] = 1.0
    return state

def _policy_action_from_grid(grid_np, greedy=False):
    policy = _load_policy()
    s = torch.from_numpy(_one_hot_from_grid(grid_np)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = policy(s).squeeze(0).cpu().numpy()
    return int(np.argmax(probs) if greedy else np.random.choice(4, p=probs))

# --- Environment step (keep your existing logic if present) ---
# Returns (next_grid, reward, done)
def step_env(grid, action):
    # grid: numpy (5,5) ints; 0 empty, 1 mouse, 2 wall, 3 trap, 4 cheese, 5 organic
    # action: 0 up, 1 down, 2 left, 3 right
    # Implements: +10 cheese/organic, -50 trap, -0.2 empty or wall bump
    H, W = grid.shape
    mouse_pos = tuple(map(int, np.argwhere(grid == 1)[0]))
    di = [-1, +1, 0, 0][action]
    dj = [0, 0, -1, +1][action]
    ni, nj = mouse_pos[0] + di, mouse_pos[1] + dj

    reward = -0.2
    done = False
    new_grid = grid.copy()

    if 0 <= ni < H and 0 <= nj < W:
        target = int(grid[ni, nj])
        if target == 2:  # wall
            # bump; stay with -0.2
            pass
        elif target == 3:  # trap
            new_grid[mouse_pos] = 0
            new_grid[ni, nj] = 1
            reward = -50.0
            done = True
        elif target in (4, 5):  # any cheese
            new_grid[mouse_pos] = 0
            new_grid[ni, nj] = 1
            reward = +10.0
            done = True
        else:  # empty
            new_grid[mouse_pos] = 0
            new_grid[ni, nj] = 1
    # else: out of bounds → wall bump: keep -0.2

    return new_grid, reward, done

# --- API endpoints ---
@csrf_exempt
def save_layout(request):
    global CURRENT_LAYOUT
    data = json.loads(request.body.decode())
    CURRENT_LAYOUT = np.array(data["grid"], dtype=np.int32)
    return JsonResponse({"ok": True})

def _policy_action_from_grid(grid_np, greedy=False, kind="trained"):
    policy = _load_policy(kind)
    s = torch.from_numpy(_one_hot_from_grid(grid_np)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = policy(s).squeeze(0).cpu().numpy()
    return int(np.argmax(probs) if greedy else np.random.choice(4, p=probs))

@csrf_exempt
def run_episode(request):
    data = json.loads(request.body.decode())
    grid = np.array(data["grid"], dtype=np.int32)

    # Params from UI (defaults are fine if not provided)
    policy_kind = data.get("policy", "random")    # "random" | "trained" | "rhlf"
    mode = data.get("mode", "stochastic")         # "stochastic" | "greedy"
    greedy = (mode == "greedy")

    if policy_kind == "random":
        action = int(np.random.randint(0, 4))   
    else:
        action = int(_policy_action_from_grid(grid, greedy=greedy, kind=policy_kind))

    # Optional: ensure exactly one mouse exists
    mice = np.argwhere(grid == 1)
    if mice.shape[0] != 1:
        return JsonResponse(
            {"error": f"Grid must contain exactly one mouse (found {mice.shape[0]})."},
            status=400
        )

    frames = [{"grid": grid.tolist(), "action": None, "reward": 0.0}]
    total_return = 0.0

    # Roll out one episode (cap ~40 steps like training)
    for _ in range(40):
        if policy_kind == "random":
            action = int(np.random.randint(0, 4))
        else:
            # stochastic vs greedy selection
            action = int(_policy_action_from_grid(grid, greedy=greedy, kind=policy_kind))

        grid, r, done = step_env(grid, action)
        total_return += r
        frames.append({"grid": grid.tolist(), "action": action, "reward": float(r)})

        if done:
            break

    # Optional server-side debug
    # print(f"episode: steps={len(frames)-1}, return={total_return:.2f}, policy={policy_kind}/{mode}")

    return JsonResponse({"frames": frames, "total_return": float(total_return)})

def _rollout_one(grid_init, policy_kind="trained", greedy=False, max_steps=40):
    grid = grid_init.copy()
    frames = [{"grid": grid.tolist(), "action": None, "reward": 0.0}]
    total_return = 0.0
    for _ in range(max_steps):
        action = (np.random.randint(0,4) if policy_kind=="random"
                  else _policy_action_from_grid(grid, greedy=greedy, kind=policy_kind))
        grid, r, done = step_env(grid, int(action))
        total_return += r
        frames.append({"grid": grid.tolist(), "action": int(action), "reward": float(r)})
        if done:
            break
    return {"frames": frames, "total_return": float(total_return)}

@csrf_exempt
def sample_pair(request):
    """
    Returns two trajectories A and B from (typically) the trained policy.
    The UI will animate both and ask the user which they prefer.
    """
    data = json.loads(request.body.decode())
    grid = np.array(data["grid"], dtype=np.int32)
    policy_kind = data.get("policy", "trained")
    mode = data.get("mode", "stochastic")
    greedy = (mode == "greedy")

    pair_id = str(uuid.uuid4())
    # Make two independent rollouts (you can also vary policy_kind per arm)
    A = _rollout_one(grid, policy_kind=policy_kind, greedy=greedy)
    B = _rollout_one(grid, policy_kind=policy_kind, greedy=greedy)

    return JsonResponse({"pair_id": pair_id, "A": A, "B": B})

def _append_pref(pair_id, trajA, trajB, winner, meta):
    rec = {
        "ts": time.time(),
        "pair_id": pair_id,
        "winner": winner,           # "A" or "B"
        "A": trajA,                 # includes frames + total_return
        "B": trajB,
        "meta": meta,               # e.g., {"policy": "...", "mode": "..."}
    }
    with open(PREF_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

@csrf_exempt
def submit_pref(request):
    """
    Body: { pair_id, winner: "A"|"B", A, B, policy, mode }
    """
    data = json.loads(request.body.decode())
    pair_id = data["pair_id"]
    winner  = data["winner"]
    trajA   = data["A"]
    trajB   = data["B"]
    policy  = data.get("policy", "trained")
    mode    = data.get("mode", "stochastic")
    _append_pref(pair_id, trajA, trajB, winner, {"policy": policy, "mode": mode})
    return JsonResponse({"ok": True})

def _fake_user_winner(trajA, trajB):
    def ate_organic(traj):
        # last non-null reward frame tells us if it ended on cheese/trap/empty; check grids for code 5 over time
        # simpler: see if any reward +10 happened on a frame where the moved-to cell was organic (5)
        for fr in traj["frames"][1:]:
            # heuristic: prefer A if B ever steps on organic (code==5) at the step it finishes
            pass
        return False
    # prefer the one that avoids organic (return "A" or "B")
    return "A"


# 1) Train/refresh reward model from preferences.jsonl (short run)
@csrf_exempt
def train_reward_now(request):
    from .train_reward import train_bt
    try:
        body = json.loads(request.body.decode() or "{}")
        jsonl  = body.get("jsonl", "preferences.jsonl")
        epochs = int(body.get("epochs", 5))
        batch  = int(body.get("batch_size", 8))
        if not os.path.exists(jsonl):
            return JsonResponse({"ok": False, "error": f"{jsonl} not found"}, status=400)
        info = train_bt(jsonl=jsonl, epochs=epochs, batch_size=batch)
        return JsonResponse({"ok": True, "model": "reward_net.pt", "metrics": info})
    except Exception as e:
        import traceback; tb = traceback.format_exc()
        print(tb)
        return JsonResponse({"ok": False, "error": str(e), "traceback": tb}, status=500)



# 2) Fine-tune policy with RLHF (short run)
@csrf_exempt
def train_rlhf_now(request):
    from .train_rlhf import main as rlhf_main
    try:
        body = json.loads(request.body.decode() or "{}")

        steps  = int(body.get("steps", 50))
        gamma  = float(body.get("gamma", 0.99))
        ent    = float(body.get("entropy", 0.0015))
        klc    = float(body.get("kl_coef", 0.05))
        msteps = int(body.get("max_steps", 40))
        ntrajs = int(body.get("trajs_per_update", 16))
        lr     = float(body.get("lr", 3e-4))

        # Resolve common locations (project root + app dir)
        base = getattr(settings, "BASE_DIR", os.getcwd())
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Preflight: base policy
        have_base = any(os.path.exists(p) for p in [
            os.path.join(base, "policy_net_best.pt"),
            os.path.join(base, "policy_net.pt"),
            os.path.join(app_dir, "policy_net_best.pt"),
            os.path.join(app_dir, "policy_net.pt"),
        ])
        if not have_base:
            return JsonResponse({"ok": False, "error": "No base policy found. Train REINFORCE first."}, status=400)

        # Preflight: reward
        have_reward = any(os.path.exists(p) for p in [
            os.path.join(base, "reward_net.pt"),
            os.path.join(app_dir, "reward_net.pt"),
        ])
        if not have_reward:
            return JsonResponse({"ok": False, "error": "reward_net.pt not found. Train Reward (BT) first."}, status=400)

        ok, info = rlhf_main(
            steps=steps, gamma=gamma, entropy=ent, kl_coef=klc,
            max_steps=msteps, trajs_per_update=ntrajs, lr=lr
        )
        info = info or {}
        metrics = {
            "learned_return_mean": float(info.get("learned_return_mean")) if isinstance(info.get("learned_return_mean"), (int,float)) else None,
            "kl_to_ref": float(info.get("kl_to_ref")) if isinstance(info.get("kl_to_ref"), (int,float)) else None,
        }
        return JsonResponse({"ok": bool(ok), "ckpt": info.get("ckpt") or "policy_net_rlhf.pt", "metrics": metrics})

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JsonResponse({"ok": False, "error": str(e), "traceback": tb}, status=500)


@csrf_exempt
def train_reinforce_now(request):
    """
    Minimal REINFORCE training loop.
    Body (optional): { "updates": 10, "batch_episodes": 32, "gamma": 0.99, "entropy": 0.01, "max_steps": 40 }
    Saves to: policy_net.pt (and a best checkpoint policy_net_best.pt)
    """
    import json, numpy as np, torch
    import torch.optim as optim
    from torch.distributions import Categorical
    from .policy_net import PolicyNet

    body = json.loads(request.body.decode() or "{}")
    updates = int(body.get("updates", 10))
    batch_episodes = int(body.get("batch_episodes", 32))
    gamma = float(body.get("gamma", 0.99))
    entropy_coef = float(body.get("entropy", 1e-3))
    max_steps = int(body.get("max_steps", 40))

    def onehot(grid_np):
        s = np.zeros((6, 5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                s[grid_np[i, j], i, j] = 1.0
        return s

    # default starter layout if none saved from UI
    grid0 = (CURRENT_LAYOUT if isinstance(CURRENT_LAYOUT, np.ndarray) else
             np.array([[0,0,0,0,0],[0,2,0,3,0],[0,0,1,0,0],[0,4,0,5,0],[0,0,0,0,0]], dtype=np.int32))

    device = DEVICE
    policy = PolicyNet().to(device)
    # continue training if an old checkpoint exists
    for cand in ["policy_net_best.pt", "policy_net.pt"]:
        if os.path.exists(cand):
            policy.load_state_dict(torch.load(cand, map_location=device))
            break
    opt = optim.Adam(policy.parameters(), lr=3e-4)

    def rollout_once(grid_init):
        logps, entrs, rewards = [], [], []
        grid = grid_init.copy()
        for _ in range(max_steps):
            s = torch.from_numpy(onehot(grid)).unsqueeze(0).to(device)
            probs = policy(s).squeeze(0)
            dist = Categorical(probs)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            entrs.append(dist.entropy())
            grid, r, done = step_env(grid, int(a.item()))
            rewards.append(r)
            if done: break
        return logps, entrs, rewards

    def discount(rs):
        G, out = 0.0, []
        for r in reversed(rs):
            G = r + gamma * G
            out.append(G)
        out.reverse()
        return out

    best = -1e9
    stats = []
    for u in range(1, updates+1):
        trajs, allG = [], []
        for _ in range(batch_episodes):
            logps, entrs, rews = rollout_once(grid0)
            trajs.append((logps, entrs, rews))
            allG.append(sum(rews))

        # policy gradient step
        log_all, ent_all, ret_all = [], [], []
        for logps, entrs, rews in trajs:
            Gs = discount(rews)
            log_all.extend(logps)
            ent_all.extend(entrs)
            ret_all.extend(Gs)

        log_all = torch.stack(log_all).to(device)
        ent_all = torch.stack(ent_all).to(device)
        ret_all = torch.tensor(ret_all, dtype=torch.float32, device=device)
        # simple baseline: subtract mean
        ret_all = ret_all - ret_all.mean()

        loss = -(log_all * ret_all).mean() - entropy_coef * ent_all.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        avg_return = float(np.mean(allG))
        stats.append({"update": u, "avg_return": round(avg_return,2), "loss": round(float(loss.item()),4)})
        if avg_return > best:
            best = avg_return
            torch.save(policy.state_dict(), "policy_net_best.pt")

    torch.save(policy.state_dict(), "policy_net.pt")
    # invalidate cached policies so future /run-episode uses the fresh weights
    global _POLICY
    _POLICY = {}
    return JsonResponse({"ok": True, "stats": stats, "saved": ["policy_net_best.pt","policy_net.pt"]})

@csrf_exempt
def reset_training(request):
    """
    Deletes trained/rlhf/reward checkpoints and clears preferences.jsonl.
    """
    targets = [
        "policy_net.pt", "policy_net_best.pt",
        "policy_net_rlhf.pt", "policy_net_rlhf_best.pt",
        "reward_net.pt", "preferences.jsonl"
    ]
    deleted = []
    for p in targets:
        try:
            os.remove(p); deleted.append(p)
        except FileNotFoundError:
            pass
    # clear cached policies
    global _POLICY
    _POLICY = {}
    return JsonResponse({"ok": True, "deleted": deleted})


@csrf_exempt
def clear_preferences(request):
    """
    Delete preferences.jsonl only.
    Returns: { ok: true, deleted: ["preferences.jsonl"] } (or [] if not present)
    """
    try:
        if request.method != "POST":
            return JsonResponse({"ok": False, "error": "POST required"}, status=405)

        deleted = []
        path_ = "preferences.jsonl"
        try:
            os.remove(path_)
            deleted.append(path_)
        except FileNotFoundError:
            pass

        return JsonResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JsonResponse({"ok": False, "error": str(e), "traceback": tb}, status=500)


# --- Allowed model files and their "kinds" (prevents arbitrary file access)
ALLOWED_MODELS = {
    "policy_trained": ["policy_net_best.pt", "policy_net.pt"],
    "policy_rlhf":    ["policy_net_rlhf_best.pt", "policy_net_rlhf.pt"],
    "reward":         ["reward_net.pt"],
}

def _find_first_existing(filename_list):
    """Search in BASE_DIR and app dir; return absolute path or None."""
    base = getattr(settings, "BASE_DIR", os.getcwd())
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in filename_list:
        for root in (base, app_dir):
            path = os.path.join(root, fname)
            if os.path.exists(path):
                return path, fname
    return None, None

def _octet_response(abs_path, download_name):
    return FileResponse(open(abs_path, "rb"),
                        as_attachment=True,
                        filename=download_name,
                        content_type="application/octet-stream")

# GET /project5/download-model/<kind>/
def download_model(request, kind: str):
    """Download a single model by kind: policy_trained | policy_rlhf | reward."""
    candidates = ALLOWED_MODELS.get(kind)
    if not candidates:
        raise Http404("Unknown model kind.")
    abs_path, fname = _find_first_existing(candidates)
    if not abs_path:
        raise Http404("Model not found. Train it first.")
    return _octet_response(abs_path, fname)

# GET /project5/download-models-zip/
def download_models_zip(request):
    """Zip up all existing models and return as one download."""
    base = getattr(settings, "BASE_DIR", os.getcwd())
    app_dir = os.path.dirname(os.path.abspath(__file__))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        added = False
        for kind, names in ALLOWED_MODELS.items():
            for name in names:
                for root in (base, app_dir):
                    path = os.path.join(root, name)
                    if os.path.exists(path):
                        zf.write(path, arcname=name)
                        added = True
                        break
    if not added:
        raise Http404("No model files found to download.")
    buf.seek(0)
    return FileResponse(buf,
                        as_attachment=True,
                        filename="project5_models.zip",
                        content_type="application/zip")


