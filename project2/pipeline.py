import os
import re
import pickle
import numpy as np
import pandas as pd

# --- NLTK (no downloads at import) ---
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Prefer a local nltk_data folder inside this app (project2/nltk_data)
_APP_DIR = os.path.dirname(__file__)
_LOCAL_NLTK = os.path.join(_APP_DIR, "nltk_data")
if os.path.isdir(_LOCAL_NLTK):
    if _LOCAL_NLTK not in nltk.data.path:
        nltk.data.path.insert(0, _LOCAL_NLTK)

def _ensure_nltk_local_or_fail():
    """
    Ensure required NLTK resources exist locally.
    We DO NOT download here. If missing, raise with a clear message.
    Required:
      - tokenizers/punkt
      - corpora/stopwords
      - corpora/wordnet
      - taggers/averaged_perceptron_tagger
    """
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ]
    missing = []
    for res_path, _name in required:
        try:
            nltk.data.find(res_path)
        except LookupError:
            missing.append(res_path)

    if missing:
        msg = (
            "Missing NLTK data locally: "
            + ", ".join(missing)
            + "\nPlease place these inside: "
            + _LOCAL_NLTK
            + "\n\nHow to fetch once on your machine:\n"
            + ">>> import nltk\n"
            + f">>> nltk.download('punkt', download_dir=r'{_LOCAL_NLTK}')\n"
            + f">>> nltk.download('stopwords', download_dir=r'{_LOCAL_NLTK}')\n"
            + f">>> nltk.download('wordnet', download_dir=r'{_LOCAL_NLTK}')\n"
            + f">>> nltk.download('averaged_perceptron_tagger', download_dir=r'{_LOCAL_NLTK}')\n"
        )
        raise RuntimeError(msg)

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Data loading + preprocessing
# -----------------------------
class IMDBDataLoader:
    def __init__(self, filedirectory, filename):
        self.filedirectory = filedirectory
        self.filepath = os.path.join(filedirectory, filename)
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess(self):
        # Ensure NLTK resources exist locally (no downloads here)
        _ensure_nltk_local_or_fail()

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        preprocessed_texts = []
        for text in self.df["review"]:
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            tokens = word_tokenize(text)
            tagged_tokens = pos_tag(tokens)
            cleaned_tokens = [
                lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
                for word, pos in tagged_tokens
                if word not in stop_words and len(word) > 2
            ]
            cleaned_text = " ".join(cleaned_tokens)
            preprocessed_texts.append(cleaned_text)

        self.df["clean_review"] = preprocessed_texts

    def save_preprocessed_data(self, output_filename="imdb_dataset_preprocessed.csv"):
        os.makedirs(self.filedirectory, exist_ok=True)
        output_path = os.path.join(self.filedirectory, output_filename)
        self.df.to_csv(output_path, index=False)


# -----------------------------
# Supervised model (Task 1)
# -----------------------------
class SVMTextClassifier:
    def __init__(self, model=None, vectorizer=None, label_encoder=None):
        self.model = model if model else LinearSVC()
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.confusion_matrix_ = None
        self.report = None

        # Cached UI data (so we don’t open the CSV once trained)
        self.cached_sample_reviews = None
        self.cached_total_reviews = None
        self.cached_num_pos = None
        self.cached_num_neg = None

    def train(self, df):
        y = self.label_encoder.fit_transform(df['sentiment'])
        X = self.vectorizer.fit_transform(df['clean_review'])

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, model=None, X_test=None, y_test=None):
        if model is None:
            model = self.model
        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test

        if model is None or X_test is None or y_test is None:
            raise ValueError("Model and test data must be provided or trained before evaluation.")

        self.y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.y_pred)
        self.confusion_matrix_ = confusion_matrix(y_test, self.y_pred)
        self.report = classification_report(
            y_test, self.y_pred, target_names=self.label_encoder.classes_, output_dict=True
        )

        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix_,
            "report": self.report
        }

    def save_model(self, path):
        payload = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'accuracy': self.accuracy,
            'confusion_matrix': self.confusion_matrix_,
            'report': self.report,
            'cached_sample_reviews': self.cached_sample_reviews,
            'cached_total_reviews': self.cached_total_reviews,
            'cached_num_pos': self.cached_num_pos,
            'cached_num_neg': self.cached_num_neg,
            'cached_avg_tokens': getattr(self, 'cached_avg_tokens', None),
            'cached_min_tokens': getattr(self, 'cached_min_tokens', None),
            'cached_max_tokens': getattr(self, 'cached_max_tokens', None),
            'cached_vocab_size': getattr(self, 'cached_vocab_size', None),
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            model=data['model'],
            vectorizer=data['vectorizer'],
            label_encoder=data['label_encoder']
        )
        instance.accuracy = data.get('accuracy')
        instance.confusion_matrix_ = data.get('confusion_matrix')
        instance.report = data.get('report')
        instance.cached_sample_reviews = data.get('cached_sample_reviews')
        instance.cached_total_reviews = data.get('cached_total_reviews')
        instance.cached_num_pos = data.get('cached_num_pos')
        instance.cached_num_neg = data.get('cached_num_neg')
        instance.cached_avg_tokens = data.get('cached_avg_tokens')
        instance.cached_min_tokens = data.get('cached_min_tokens')
        instance.cached_max_tokens = data.get('cached_max_tokens')
        instance.cached_vocab_size = data.get('cached_vocab_size')
        return instance
    
    def _fit_vectorizer_and_encoder(self, df):
        texts = df['clean_review'].astype(str).tolist()
        self.vectorizer.fit(texts)
        labels = df['sentiment'].astype(str).tolist()
        self.label_encoder.fit(labels)

    def init_with_test_split(self, df, test_size=0.2, random_state=42):
        """
        Initialize vectorizer, label encoder, and create a test split,
        but don't train the classifier yet.
        """
        from sklearn.model_selection import train_test_split

        # fit vectorizer & encoder on all data
        texts = df['clean_review'].astype(str).tolist()
        self.vectorizer.fit(texts)
        labels = df['sentiment'].astype(str).tolist()
        self.label_encoder.fit(labels)

        # make a fixed test split for evaluation
        X = self.vectorizer.transform(texts)
        y = self.label_encoder.transform(labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )


# ------------------------------------
# Active Learning (Tasks 2 & 3)
# ------------------------------------
class ActiveLearningSession:
    """
    Pool-based Active Learning with simulated oracle.
    Strategies: 'least_confidence', 'margin', 'entropy', 'random'
    """
    def __init__(self, df, max_features=5000, random_state=42):
        self.df = df.reset_index(drop=True)
        self.random_state = random_state
        # Encode labels
        self.le = LabelEncoder()
        self.y_all = self.le.fit_transform(self.df['sentiment'])
        # Representation
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.X_all = self.vectorizer.fit_transform(self.df['clean_review'])
        # Probabilistic model (calibrated SVM for predict_proba)
        base = LinearSVC()
        self.model = CalibratedClassifierCV(base, method='sigmoid', cv=3)

        # Hold out a stratified test set for fair evaluation across AL budget
        self.train_idx, self.test_idx = train_test_split(
            np.arange(self.X_all.shape[0]),
            test_size=0.2, random_state=random_state, stratify=self.y_all
        )

    def _init_labeled_seed(self, n_start=20):
        rng = np.random.default_rng(self.random_state)
        train_idx = self.train_idx.copy()
        y = self.y_all[train_idx]
        labeled = []

        for cls in np.unique(y):
            cls_idx = train_idx[y == cls]
            take = min(n_start // 2, len(cls_idx))
            if take > 0:
                labeled.extend(rng.choice(cls_idx, size=take, replace=False))

        if len(labeled) < n_start:
            pool = np.setdiff1d(train_idx, np.array(labeled, dtype=int))
            extra = n_start - len(labeled)
            if extra > 0 and len(pool) > 0:
                labeled.extend(rng.choice(pool, size=min(extra, len(pool)), replace=False))

        labeled = np.array(sorted(set(labeled)), dtype=int)
        unlabeled = np.setdiff1d(train_idx, labeled)
        return labeled, unlabeled

    def _utilities(self, strategy, proba):
        if strategy == 'least_confidence':
            util = 1.0 - np.max(proba, axis=1)
            order = np.argsort(-util)  # high util first
        elif strategy == 'margin':
            part = -np.partition(-proba, 1, axis=1)
            margins = part[:, 0] - part[:, 1]
            order = np.argsort(margins)  # smallest margin first
        elif strategy == 'entropy':
            eps = 1e-12
            ent = -np.sum(proba * np.log(proba + eps), axis=1)
            order = np.argsort(-ent)  # highest entropy first
        elif strategy == 'random':
            order = np.arange(proba.shape[0])
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(order)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return order

    def run(self, strategy='least_confidence', budget=200, batch_size=20, n_start=20):
        labeled, unlabeled = self._init_labeled_seed(n_start=n_start)

        X = self.X_all
        y = self.y_all
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        queried = 0
        curve = []

        self.model.fit(X[labeled], y[labeled])
        acc = accuracy_score(y_test, self.model.predict(X_test))
        curve.append({'labeled': int(len(labeled)), 'accuracy': float(acc)})

        while queried < budget and len(unlabeled) > 0:
            proba = self.model.predict_proba(X[unlabeled])
            order = self._utilities(strategy=strategy, proba=proba)

            k = min(batch_size, len(unlabeled), budget - queried)
            take_idx = unlabeled[order[:k]]

            labeled = np.concatenate([labeled, take_idx])
            unlabeled = np.setdiff1d(unlabeled, take_idx)

            self.model.fit(X[labeled], y[labeled])
            acc = accuracy_score(y_test, self.model.predict(X_test))
            curve.append({'labeled': int(len(labeled)), 'accuracy': float(acc)})

            queried += k

        return {
            'strategy': strategy,
            'budget': int(budget),
            'batch_size': int(batch_size),
            'curve': curve
        }

# ------------------------------------
# Human-in-the-loop Active Learning (Task 3: simple labeling UI)
# ------------------------------------
class HumanALSession:
    """
    Manages a pool-based AL session where labels come from a human (UI).
    - Uses the same TF-IDF + Linear SVM (calibrated) as Task 1/2.
    - Keeps a train/test split; training set is gradually labeled by the user.
    - Evaluates accuracy on the fixed test split after each (possible) retrain.
    """
    def __init__(self, df, max_features=5000, random_state=42):
        self.df = df.reset_index(drop=True)
        self.random_state = random_state

        # Encode labels (used only for evaluation; training uses user-provided labels)
        self.le = LabelEncoder()
        self.y_all = self.le.fit_transform(self.df['sentiment'])

        # Representation
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.X_all = self.vectorizer.fit_transform(self.df['clean_review'])

        # Probabilistic model
        base = LinearSVC()
        self.model = CalibratedClassifierCV(base, method='sigmoid', cv=3)

        # Fixed test split (for consistent scoring)
        idx = np.arange(self.X_all.shape[0])
        self.train_idx, self.test_idx = train_test_split(
            idx, test_size=0.2, random_state=random_state, stratify=self.y_all
        )

        # Session state
        self.labeled = np.array([], dtype=int)      # indices with user labels
        self.unlabeled = self.train_idx.copy()      # remaining pool
        self.y_labels = {}                          # idx -> encoded label int
        self.accuracy = None                        # last test accuracy (float)

    # ---------- utilities (same scoring rules as your AL session) ----------
    def _utilities(self, strategy, proba):
        if strategy == 'least_confidence':
            util = 1.0 - np.max(proba, axis=1)
            order = np.argsort(-util)
        elif strategy == 'margin':
            part = -np.partition(-proba, 1, axis=1)
            margins = part[:, 0] - part[:, 1]
            order = np.argsort(margins)  # smallest margin first
        elif strategy == 'entropy':
            eps = 1e-12
            ent = -np.sum(proba * np.log(proba + eps), axis=1)
            order = np.argsort(-ent)
        elif strategy == 'random':
            order = np.arange(proba.shape[0])
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(order)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return order

    def _can_train(self):
        # need at least 2 samples and 2 classes for SVM to fit
        if len(self.labeled) < 2:
            return False
        y = np.array([self.y_labels[i] for i in self.labeled])
        return len(np.unique(y)) >= 2

    def _fit_and_eval(self):
        """Fit model if possible; update self.accuracy on the test split."""
        if not self._can_train():
            return False
        X = self.X_all[self.labeled]
        y = np.array([self.y_labels[i] for i in self.labeled])
        self.model.fit(X, y)

        X_test = self.X_all[self.test_idx]
        y_test = self.y
