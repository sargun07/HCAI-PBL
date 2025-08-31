import json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .reward_model import RewardNet
import numpy as np

class PrefPairs(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("winner") not in ("A", "B"): 
                    continue
                self.rows.append(rec)

    def _traj_to_states(self, traj):
        # extract states from frames (each frame has "grid" codes)
        S = []
        for fr in traj["frames"]:
            grid = np.array(fr["grid"], dtype=np.int64)  # (5,5)
            onehot = np.zeros((6,5,5), dtype=np.float32)
            for i in range(5):
                for j in range(5):
                    onehot[grid[i,j], i, j] = 1.0
            S.append(onehot)
        return np.stack(S,0)  # [T,6,5,5]

    def __getitem__(self, idx):
        rec = self.rows[idx]
        A = self._traj_to_states(rec["A"])
        B = self._traj_to_states(rec["B"])
        y = 1.0 if rec["winner"] == "A" else 0.0
        return torch.from_numpy(A), torch.from_numpy(B), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.rows)

# def traj_score(reward_net, states):  # states: [T,6,5,5]
#     with torch.no_grad():
#         pass
#     # compute un-discounted cumulative reward (or discounted if you prefer)
#     T = states.shape[0]
#     states = states.to(device)  # [T,6,5,5]
#     r = reward_net(states)      # [T]
#     return r.sum()              # scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bt(jsonl="preferences.jsonl", epochs=10, lr=3e-4, batch_size=8):
    ds = PrefPairs(jsonl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)
    net = RewardNet().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    last_loss = None
    for ep in range(1, epochs+1):
        total = 0.0
        for batch in dl:
            logits, labels = [], []
            for A, B, y in batch:
                A = A.float().to(device); B = B.float().to(device)
                SA = net(A).sum(); SB = net(B).sum()
                logits.append(SA - SB); labels.append(y.to(device))
            logits = torch.stack(logits); labels = torch.stack(labels)
            loss = bce(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(batch)
        last_loss = total / max(1, len(ds))
        print(f"[epoch {ep}] loss={last_loss:.4f}")

    torch.save(net.state_dict(), "reward_net.pt")
    print("Saved reward model to reward_net.pt")
    return {"loss": float(last_loss), "n_pairs": len(ds)}


if __name__ == "__main__":
    train_bt()
