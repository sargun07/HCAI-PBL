import os, uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.shortcuts import render
from django.utils import timezone
import numpy as np 

from .pipeline import IMDBDataLoader, SVMTextClassifier, ActiveLearningSession
from sklearn.metrics import classification_report, confusion_matrix

_HUMAN_AL: Dict[str, "HumanALSession"] = {}

# Where to save/load the trained model: keep it inside the app folder
_APP_DIR = os.path.dirname(__file__)
_MODEL_PKL = os.path.join(_APP_DIR, 'svm_model.pkl')

# Prefer dataset under MEDIA_ROOT/project2; fall back to app-local data/
def _get_data_base_path():
    media_path = os.path.join(getattr(settings, 'MEDIA_ROOT', ''), 'project2')
    if media_path and os.path.isdir(media_path):
        return media_path
    fallback = os.path.join(_APP_DIR, 'data')
    os.makedirs(fallback, exist_ok=True)
    return fallback

def _ensure_preprocessed_df(base_path, raw_file='IMDB Dataset.csv', pre_file='imdb_dataset_preprocessed.csv'):
    """
    Load preprocessed CSV if present, else preprocess from raw once.
    This is only called when needed (first train or cache-miss), not on every page load.
    """
    pre_path = os.path.join(base_path, pre_file)
    raw_path = os.path.join(base_path, raw_file)

    if os.path.exists(pre_path):
        return IMDBDataLoader(base_path, pre_file).load_data()

    # No preprocessed file yet – build it if we have the raw CSV
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Could not find raw dataset at: {raw_path}\n"
            "Place 'IMDB Dataset.csv' under MEDIA_ROOT/project2/ or project2/data/."
        )

    loader = IMDBDataLoader(base_path, raw_file)
    df = loader.load_data()
    loader.preprocess()
    loader.save_preprocessed_data(pre_file)
    return loader.df

def _json_safe(obj):
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def index(request):
    # FAST PATH
    if os.path.exists(_MODEL_PKL):
        clf = SVMTextClassifier.load(_MODEL_PKL)

        sample = getattr(clf, 'cached_sample_reviews', None)
        total_reviews = getattr(clf, 'cached_total_reviews', None)
        num_pos = getattr(clf, 'cached_num_pos', None)
        num_neg = getattr(clf, 'cached_num_neg', None)
        avg_len = getattr(clf, 'cached_avg_tokens', None)
        min_len = getattr(clf, 'cached_min_tokens', None)
        max_len = getattr(clf, 'cached_max_tokens', None)
        vocab_size = getattr(clf, 'cached_vocab_size', None)

        # If summary was never cached (older pickle), try to fill it cheaply from preprocessed CSV
        if sample is None or total_reviews is None:
            base_path = _get_data_base_path()
            pre_file = 'imdb_dataset_preprocessed.csv'
            pre_path = os.path.join(base_path, pre_file)
            if os.path.exists(pre_path):
                df = IMDBDataLoader(base_path, pre_file).load_data()
                sample = df.sample(10, random_state=42)[['review', 'sentiment']].to_dict(orient='records')
                total_reviews = len(df)
                num_pos = int((df['sentiment'] == 'positive').sum())
                num_neg = int((df['sentiment'] == 'negative').sum())

                token_counts = df['clean_review'].fillna('').str.split().map(len)
                avg_len = float(token_counts.mean()) if len(token_counts) else None
                min_len = int(token_counts.min()) if len(token_counts) else None
                max_len = int(token_counts.max()) if len(token_counts) else None
                vocab_size = len(getattr(clf.vectorizer, 'vocabulary_', {}) or {})

                # cache back into pickle (FIXED TYPO: total_reviews)
                clf.cached_sample_reviews = sample
                clf.cached_total_reviews = total_reviews
                clf.cached_num_pos = num_pos
                clf.cached_num_neg = num_neg
                clf.cached_avg_tokens = avg_len
                clf.cached_min_tokens = min_len
                clf.cached_max_tokens = max_len
                clf.cached_vocab_size = vocab_size
                clf.save_model(_MODEL_PKL)
            else:
                # Show metrics without dataset summary
                sample, total_reviews, num_pos, num_neg = [], None, None, None
                avg_len = min_len = max_len = vocab_size = None

        results = {
            "accuracy": clf.accuracy,
            "confusion_matrix": clf.confusion_matrix_,
            "report": clf.report
        }
        print("average lenght: ",avg_len)
        return render(request, 'project2/interface.html', {
            'sample_reviews': sample,
            'total_reviews': total_reviews,
            'num_pos': num_pos,
            'num_neg': num_neg,
            'avg_len': avg_len,
            'min_len': min_len,
            'max_len': max_len,
            'vocab_size': vocab_size,
            "accuracy": results["accuracy"],
            "report": results["report"],
            "confusion_matrix": results["confusion_matrix"],
        })  # :contentReference[oaicite:1]{index=1}

    # SLOW PATH (first time only)
    base_path = _get_data_base_path()
    df = _ensure_preprocessed_df(base_path)

    clf = SVMTextClassifier()
    clf.train(df)
    results = clf.evaluate()

    # Cache dataset summary + stats
    sample = df.sample(10, random_state=42)[['review', 'sentiment']].to_dict(orient='records')
    total_reviews = len(df)
    num_pos = int((df['sentiment'] == 'positive').sum())
    num_neg = int((df['sentiment'] == 'negative').sum())
    token_counts = df['clean_review'].fillna('').str.split().map(len)
    avg_len = float(token_counts.mean()) if len(token_counts) else None
    min_len = int(token_counts.min()) if len(token_counts) else None
    max_len = int(token_counts.max()) if len(token_counts) else None
    vocab_size = len(getattr(clf.vectorizer, 'vocabulary_', {}) or {})

    clf.cached_sample_reviews = sample
    clf.cached_total_reviews = total_reviews
    clf.cached_num_pos = num_pos
    clf.cached_num_neg = num_neg
    clf.cached_avg_tokens = avg_len
    clf.cached_min_tokens = min_len
    clf.cached_max_tokens = max_len
    clf.cached_vocab_size = vocab_size

    os.makedirs(os.path.dirname(_MODEL_PKL), exist_ok=True)
    clf.save_model(_MODEL_PKL)

    return render(request, 'project2/interface.html', {
        'sample_reviews': sample,
        'total_reviews': total_reviews,
        'num_pos': num_pos,
        'num_neg': num_neg,
        'avg_len': avg_len,
        'min_len': min_len,
        'max_len': max_len,
        'vocab_size': vocab_size,
        "accuracy": results["accuracy"],
        "report": results["report"],
        "confusion_matrix": results["confusion_matrix"],
    })  # :contentReference[oaicite:2]{index=2}

@require_POST
def preprocess_dataset(request):
    try:
        base_path = _get_data_base_path()
        loader = IMDBDataLoader(base_path, 'IMDB Dataset.csv')
        df = loader.load_data()
        loader.preprocess()
        loader.save_preprocessed_data('imdb_dataset_preprocessed.csv')
        return JsonResponse({"success": True, "rows": len(df)})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

# ---------- Train / Retrain ----------
@csrf_exempt
def train(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)
    try:
        base_path = _get_data_base_path()
        df = _ensure_preprocessed_df(base_path)

        clf = SVMTextClassifier()
        clf.train(df)
        results = clf.evaluate()  # accuracy, confusion_matrix, report (pure sklearn) :contentReference[oaicite:3]{index=3}

        # dataset summary + stats
        sample = df.sample(10, random_state=42)[['review','sentiment']].to_dict(orient='records')
        token_counts = df['clean_review'].fillna('').str.split().map(len)
        clf.cached_sample_reviews = sample
        clf.cached_total_reviews = int(len(df))
        clf.cached_num_pos = int((df['sentiment'] == 'positive').sum())
        clf.cached_num_neg = int((df['sentiment'] == 'negative').sum())
        clf.cached_avg_tokens = float(token_counts.mean()) if len(token_counts) else None
        clf.cached_min_tokens = int(token_counts.min()) if len(token_counts) else None
        clf.cached_max_tokens = int(token_counts.max()) if len(token_counts) else None
        clf.cached_vocab_size = int(len(getattr(clf.vectorizer, 'vocabulary_', {}) or {}))

        os.makedirs(os.path.dirname(_MODEL_PKL), exist_ok=True)
        if os.path.exists(_MODEL_PKL):
            os.remove(_MODEL_PKL)  # delete old pickle before saving new one
        clf.save_model(_MODEL_PKL)

        return JsonResponse({'success': True, **_json_safe(results)})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)



def download_model(request):
    if not os.path.exists(_MODEL_PKL):
        raise Http404("Model file not found. Train the model first.")
    f = open(_MODEL_PKL, 'rb')
    return FileResponse(f, as_attachment=True, filename=os.path.basename(_MODEL_PKL))

def download_dataset(request):
    base_path = _get_data_base_path()
    pre_path = os.path.join(base_path, 'imdb_dataset_preprocessed.csv')
    if not os.path.exists(pre_path):
        raise Http404("Preprocessed dataset not found. Train once to generate it.")
    f = open(pre_path, 'rb')
    return FileResponse(f, as_attachment=True, filename='imdb_preprocessed.csv')


def run_active_learning(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)
    try:
        base_path = _get_data_base_path()
        df = _ensure_preprocessed_df(base_path)

        strategy = request.POST.get('strategy', 'least_confidence')
        budget = int(request.POST.get('budget', '200'))
        batch_size = int(request.POST.get('batch_size', '20'))

        al = ActiveLearningSession(df=df, max_features=5000, random_state=42)
        output = al.run(strategy=strategy, budget=budget, batch_size=batch_size, n_start=20)

        return JsonResponse({'success': True, **output})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

# ----- Utility: get or make a session key -----
def _user_session_key(request):
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key

# ===== Human-in-the-loop Active Learning =====
@dataclass
class HumanALSession:
    strategy: str
    budget: int
    created_at: str
    df: "pd.DataFrame" = field(repr=False)
    clf: "SVMTextClassifier" = field(repr=False)
    labeled_idx: List[int] = field(default_factory=list)
    pool_idx: List[int] = field(default_factory=list)
    used_budget: int = 0
    accuracy_history: List[float] = field(default_factory=list)
    labeled_count_history: List[int] = field(default_factory=list)
    finished: bool = False
    last_candidate_id: Optional[int] = None
    model_trained: bool = False  # NEW

    def _ensure_initialized(self):
        if not self.pool_idx:
            self.pool_idx = list(range(len(self.df)))

    def _vectorize(self, texts):
        return self.clf.vectorizer.transform(texts)

    def _fit_if_possible(self):
        """Fit only when we have at least two classes labeled."""
        import numpy as _np
        if len(self.labeled_idx) == 0:
            self.model_trained = False
            return False

        # Get labels for the currently labeled indices
        y_str = self.df.loc[self.labeled_idx, 'sentiment'].astype(str).tolist()
        y_enc = self.clf.label_encoder.transform(y_str)
        uniq = _np.unique(y_enc)
        if uniq.size < 2:
            # not enough classes yet
            self.model_trained = False
            return False

        X = self._vectorize(self.df.loc[self.labeled_idx, 'clean_review'])
        self.clf.model.fit(X, y_enc)
        self.model_trained = True
        return True

    def _score(self):
        if not self.model_trained:
            self.accuracy_history.append(None)
            self.labeled_count_history.append(len(self.labeled_idx))
            return None
        try:
            res = self.clf.evaluate()
            acc = float(res.get('accuracy', 0.0))
        except Exception:
            acc = None
        self.accuracy_history.append(acc)
        self.labeled_count_history.append(len(self.labeled_idx))
        return acc


    def _pick_candidate(self):
        """Pick one index from pool; random until the model is trained."""
        import numpy as _np
        if not self.pool_idx:
            return None

        # If we cannot or have not trained yet, just pick random to diversify labels
        if not self.model_trained:
            return _np.random.choice(self.pool_idx)

        # Compute uncertainty scores on a slice of the pool
        candidate_ids = self.pool_idx[:2000] if len(self.pool_idx) > 2000 else self.pool_idx
        texts = self.df.loc[candidate_ids, 'clean_review'].tolist()
        Xp = self._vectorize(texts)

        try:
            probs = self.clf.model.predict_proba(Xp)
            if self.strategy == 'least_confidence':
                conf = probs.max(axis=1)
                scores = 1.0 - conf
            elif self.strategy == 'margin':
                part = -np.partition(probs, -2, axis=1)[:, -2:]
                margins = (part[:,1] - part[:,0])
                scores = -margins
            elif self.strategy == 'entropy':
                with np.errstate(divide='ignore', invalid='ignore'):
                    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
                scores = entropy
            else:
                return np.random.choice(candidate_ids)
        except Exception:
            # fallback: decision_function
            try:
                f = self.clf.model.decision_function(Xp)
                if f.ndim == 1:
                    margin = np.abs(f)
                    scores = -margin
                else:
                    part = -np.partition(f, -2, axis=1)[:, -2:]
                    margin = (part[:,0] - part[:,1])
                    scores = -margin
            except Exception:
                return np.random.choice(candidate_ids)

        best_local = int(np.argmax(scores))
        return candidate_ids[best_local]

    def next_item(self):
        self._ensure_initialized()
        if self.finished or self.used_budget >= self.budget:
            self.finished = True
            return None
        cand = self._pick_candidate()
        self.last_candidate_id = cand
        row = self.df.loc[cand]
        return {"id": int(cand), "review": str(row['review'])}

    def submit_label(self, item_id: int, label: str):
        if self.finished:
            return {"finished": True, "message": "Budget exhausted."}
        if self.last_candidate_id is None or int(item_id) != int(self.last_candidate_id):
            return {"finished": False, "message": "Mismatched item id."}

        # Move from pool to labeled set, store human label
        if item_id in self.pool_idx:
            self.pool_idx.remove(item_id)
        self.labeled_idx.append(item_id)
        self.df.loc[item_id, 'sentiment'] = label

        # Try to fit (only if ≥2 classes); then score
        fitted = self._fit_if_possible()
        acc = self._score()  # None if not fitted yet

        self.used_budget += 1
        if self.used_budget >= self.budget:
            self.finished = True

        return {
            "finished": self.finished,
            "used_budget": int(self.used_budget),
            "labeled_so_far": int(len(self.labeled_idx)),
            "accuracy": (None if acc is None else float(acc)),
            "note": (None if fitted else "Need at least one sample from each class to train; picking randomly until then.")
        }


@require_POST
def al_human_init(request):
    """
    Initialize a human-in-the-loop AL session.
    POST fields: strategy (least_confidence|margin|entropy|random), budget (int)
    """
    try:
        strategy = request.POST.get('strategy', 'least_confidence').strip()
        budget = int(request.POST.get('budget', '50'))
        if budget <= 0: budget = 50

        # prepare dataset and a *fresh* classifier (not the pre-trained one)
        base_path = _get_data_base_path()
        df = _ensure_preprocessed_df(base_path).copy()

        clf = SVMTextClassifier()
        clf.init_with_test_split(df)

        sid = _user_session_key(request)
        _HUMAN_AL[sid] = HumanALSession(
            strategy=strategy,
            budget=budget,
            created_at=str(timezone.now()),
            df=df,
            clf=clf
        )
        return JsonResponse({"success": True, "message": "Human AL initialized."})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@require_GET
def al_human_next(request):
    """Return the next most-informative unlabeled review."""
    sid = _user_session_key(request)
    sess = _HUMAN_AL.get(sid)
    if not sess:
        return JsonResponse({"success": False, "error": "Session not initialized."}, status=400)
    item = sess.next_item()
    if item is None:
        return JsonResponse({"success": True, "finished": True})
    return JsonResponse({"success": True, "finished": False, "item": _json_safe(item)})


@require_POST
def al_human_label(request):
    """
    Accept a human label for the last shown item.
    POST fields: id (int row index), label ('positive'|'negative')
    """
    try:
        sid = _user_session_key(request)
        sess = _HUMAN_AL.get(sid)
        if not sess:
            return JsonResponse({"success": False, "error": "Session not initialized."}, status=400)

        item_id = int(request.POST.get('id'))
        label = request.POST.get('label', '').strip().lower()
        if label not in ('positive', 'negative'):
            return JsonResponse({"success": False, "error": "Label must be 'positive' or 'negative'."}, status=400)

        out = sess.submit_label(item_id, label)
        return JsonResponse({"success": True, **_json_safe(out)})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@require_POST
def al_human_reset(request):
    """Clear the human AL session."""
    sid = _user_session_key(request)
    if sid in _HUMAN_AL:
        del _HUMAN_AL[sid]
    return JsonResponse({"success": True, "message": "Human AL session cleared."})


