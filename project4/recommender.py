# project4/recommender.py
import os
import numpy as np
import pandas as pd

class MFRecommender:
    def __init__(self, data_root=None, K=20, lambda_reg=0.1, random_state=42):
        self.K = int(K)
        self.lambda_reg = float(lambda_reg)
        self.random_state = int(random_state)
        self._rng = np.random.RandomState(self.random_state)

        # Will be populated after load/train
        self.user_ids = []
        self.item_ids = []
        self.item_titles = {}
        self.U = None
        self.V = None

        # index caches
        self._iid2ix_cache = None
        self.item_ids_set = None

        # load + factorize
        if data_root is None:
            # default to same directory as this file
            data_root = os.path.dirname(os.path.abspath(__file__))
        self._load_movielens_and_train(data_root)

        # ensure maps now (prevents solve_user_vector crashes)
        self._ensure_index_maps()

    # ------------- utils for indices -------------
    def _ensure_index_maps(self):
        if self._iid2ix_cache is None:
            self._iid2ix_cache = {m: j for j, m in enumerate(self.item_ids)}
        if self.item_ids_set is None:
            self.item_ids_set = set(self.item_ids)

    @property
    def iid2ix(self):
        self._ensure_index_maps()
        return self._iid2ix_cache

    # ------------- data loading & training -------------
    def _load_movielens_and_train(self, data_root):
        """
        Accept both:
        - data_root = .../ml-100k   (contains u.data, u.item)
        - data_root = .../          (contains ml-100k/u.data, ml-100k/u.item)
        - same for ml-latest-small
        """
        roots = []
        if data_root:
            roots.append(data_root)                     # direct
            roots.append(os.path.dirname(data_root))    # parent
        roots += [
            os.getcwd(),
            os.path.join(os.getcwd(), "data"),
            os.path.expanduser("~/data"),
        ]

        found = None
        for root in roots:
            if not root or not os.path.isdir(root):
                continue

            # --- 100k: direct folder ---
            udata = os.path.join(root, "u.data")
            uitem = os.path.join(root, "u.item")
            if os.path.exists(udata) and os.path.exists(uitem) and os.path.basename(root).lower() == "ml-100k":
                found = ("ml-100k", udata, uitem)
                break

            # --- 100k: parent folder ---
            udata = os.path.join(root, "ml-100k", "u.data")
            uitem = os.path.join(root, "ml-100k", "u.item")
            if os.path.exists(udata) and os.path.exists(uitem):
                found = ("ml-100k", udata, uitem)
                break

            # --- latest-small: direct folder ---
            ratings_csv = os.path.join(root, "ratings.csv")
            movies_csv  = os.path.join(root, "movies.csv")
            if (os.path.exists(ratings_csv) and os.path.exists(movies_csv)
                    and os.path.basename(root).lower() == "ml-latest-small"):
                found = ("ml-latest-small", ratings_csv, movies_csv)
                break

            # --- latest-small: parent folder ---
            ratings_csv = os.path.join(root, "ml-latest-small", "ratings.csv")
            movies_csv  = os.path.join(root, "ml-latest-small", "movies.csv")
            if os.path.exists(ratings_csv) and os.path.exists(movies_csv):
                found = ("ml-latest-small", ratings_csv, movies_csv)
                break

        if not found:
            raise FileNotFoundError(
                "Could not find MovieLens data. Place either `ml-100k/u.data` + `ml-100k/u.item` "
                "or `ml-latest-small/ratings.csv` + `ml-latest-small/movies.csv` under a searched root, "
                "or provide a correct data_root."
            )

        kind, p1, p2 = found
        if kind == "ml-100k":
            ratings = pd.read_csv(p1, sep=r"\t", names=["userId","movieId","rating","timestamp"], engine="python")
            movies  = pd.read_csv(p2, sep="|", header=None, encoding="latin-1", engine="python")[[0,1]]
            movies.columns = ["movieId","title"]
        else:
            ratings = pd.read_csv(p1)
            movies  = pd.read_csv(p2)

        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)

        users = sorted(ratings["userId"].unique().tolist())
        items = sorted(ratings["movieId"].unique().tolist())
        uid2ix = {u:i for i,u in enumerate(users)}
        iid2ix = {m:j for j,m in enumerate(items)}
        self.user_ids = users
        self.item_ids = items
        title_map = dict(zip(movies["movieId"], movies["title"]))
        self.item_titles = {m: title_map.get(m, f"Movie {m}") for m in items}

        nU, nI = len(users), len(items)
        R = np.zeros((nU, nI), dtype=np.float32)
        for (u,m,r,*_) in ratings.itertuples(index=False):
            R[uid2ix[u], iid2ix[m]] = float(r)

        # center by user mean on observed entries
        for i in range(nU):
            mask = R[i,:] > 0
            if mask.any():
                mu = R[i,mask].mean()
                R[i,mask] -= mu

        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        K = min(self.K, len(s))
        U_k = U[:, :K]
        S_k = np.diag(np.sqrt(s[:K] + 1e-8))
        V_k = Vt[:K, :].T

        self.U = U_k @ S_k
        self.V = V_k @ S_k

        # Build lookup caches now to avoid later AttributeError
        self._iid2ix_cache = {m:j for j,m in enumerate(self.item_ids)}
        self.item_ids_set = set(self.item_ids)

        print(f"[MFRecommender] Loaded {kind}: users={nU}, items={nI}, K={K}")


    # ------------- new user solve -------------
    def solve_user_vector(self, ratings_dict):
        """
        Ridge solve: u = (V_J^T V_J + λI)^-1 V_J^T r
        ratings_dict: {movieId: rating}
        """
        self._ensure_index_maps()
        if not ratings_dict:
            return np.zeros(self.V.shape[1], dtype=np.float64)

        js, rs = [], []
        for m, r in ratings_dict.items():
            try:
                m_int = int(m)
            except Exception:
                m_int = m
            if m_int in self.item_ids_set:
                js.append(self.iid2ix[m_int])
                rs.append(float(r))

        if not js:
            return np.zeros(self.V.shape[1], dtype=np.float64)

        VJ = self.V[np.array(js, dtype=int), :]  # (lenJ, K)
        r  = np.array(rs, dtype=np.float64)      # (lenJ,)
        K  = self.V.shape[1]
        A  = VJ.T @ VJ + self.lambda_reg * np.eye(K)
        b  = VJ.T @ r
        u  = np.linalg.solve(A, b)
        return u

    # ------------- scoring -------------
    def top_n(self, user_vec, exclude_ids=None, n=10):
        self._ensure_index_maps()
        scores = user_vec @ self.V.T  # (n_items,)

        if exclude_ids:
            # allow ints or strings
            excl_ix = []
            for mid in exclude_ids:
                try:
                    mid = int(mid)
                except Exception:
                    pass
                if mid in self.item_ids_set:
                    excl_ix.append(self.iid2ix[mid])
            if excl_ix:
                scores[np.array(excl_ix, dtype=int)] = -1e9

        n = max(0, min(n, scores.shape[0]))
        if n == 0:
            return []
        idx = np.argpartition(-scores, range(n))[:n]
        idx = idx[np.argsort(-scores[idx])]

        out = []
        for j in idx:
            mid = self.item_ids[j]
            out.append((mid, self.item_titles.get(mid, f"Movie {mid}"), float(scores[j])))
        return out

    # ------------- active selection with what-if -------------
    def select_next_item(self, session_ratings, asked_set, pool_size=50, preview_low=0.5, preview_high=5.0):
        self._ensure_index_maps()
        asked_set = set(int(a) for a in asked_set)
        seen = set(int(k) for k in session_ratings.keys())

        # unseen candidates
        all_unseen = [int(m) for m in self.item_ids if (m not in seen and m not in asked_set)]
        if not all_unseen:
            return None

        # sample pool
        if len(all_unseen) > pool_size:
            pool = list(self._rng.choice(all_unseen, size=pool_size, replace=False))
        else:
            pool = all_unseen

        base_exclude = seen.copy()
        best_m = None
        best_dist = -1.0
        best_effect = 0.0  # ✅ initialize effect
        best_previews = {"low": [], "high": []}

        for c in pool:
            c = int(c)

            # low
            r_low = dict(session_ratings)
            r_low[c] = preview_low
            u_low = self.solve_user_vector(r_low)
            top_low = self.top_n(u_low, exclude_ids=base_exclude | {c}, n=10)
            set_low = {int(t[0]) for t in top_low}

            # high
            r_high = dict(session_ratings)
            r_high[c] = preview_high
            u_high = self.solve_user_vector(r_high)
            top_high = self.top_n(u_high, exclude_ids=base_exclude | {c}, n=10)
            set_high = {int(t[0]) for t in top_high}

            # Jaccard distance
            union = set_low | set_high
            inter = set_low & set_high
            dist = 1.0 - (len(inter) / max(1, len(union)))

            if dist > best_dist:
                best_dist = dist
                best_effect = dist          # ✅ keep the effect of the best candidate
                best_m = c
                # preview top-5 (by id order already in sets → cast to list)
                best_previews = {
                    "low":  [(self.item_titles[m], int(m)) for m in list(set_low)[:5]],
                    "high": [(self.item_titles[m], int(m)) for m in list(set_high)[:5]],
                }

        if best_m is None:
            best_m = int(self._rng.choice(all_unseen))

        return {
            "movieId": int(best_m),
            "title": self.item_titles.get(best_m, f"Movie {best_m}"),
            "what_if": best_previews,
            "effect": float(best_effect),   # ✅ now always defined and matches best candidate
        }
