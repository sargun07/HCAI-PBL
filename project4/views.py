# project4/views.py
import os
import csv
import datetime
import json
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.utils.crypto import get_random_string

from .recommender import MFRecommender

# -----------------------
# Model (singleton)
# -----------------------
# If needed, pass data_root=... where your MovieLens lives

DATA_ROOT = os.path.join(settings.BASE_DIR, "project4", "ml-100k")

MODEL = None
MODEL_LOAD_ERROR = None

try:
    # Adjust data_root if you keep MovieLens somewhere specific.
    MODEL = MFRecommender(data_root=DATA_ROOT, K=20, lambda_reg=0.2)
except Exception as e:
    MODEL_LOAD_ERROR = str(e)

# -----------------------
# Helpers
# -----------------------
def _ensure_session_defaults(request):
    if "participant_id" not in request.session:
        request.session["participant_id"] = get_random_string(10)
    request.session.setdefault("consented", False)
    request.session.setdefault("group", None)  # "control" or "treatment"
    request.session.setdefault("ratings", {})  # {movieId: rating}
    request.session.setdefault("asked", [])    # [movieId,...]
    request.session.modified = True

def _json_error(msg, code=400):
    return JsonResponse({"ok": False, "error": msg}, status=code)

# -----------------------
# Pages
# -----------------------
def landing(request):
    """
    Landing page: (1) PDF link (Task 1+2 write-up), (2) Start Study button (Task 3)
    """
    return render(request, "project4/landing.html", {
        "model_error": MODEL_LOAD_ERROR
    })

def index(request):
    """
    Participant interface (study UI)
    """
    _ensure_session_defaults(request)
    return render(request, "project4/interface.html", {
        "group": request.session.get("group"),
        "consented": request.session.get("consented"),
        "model_error": MODEL_LOAD_ERROR
    })

# -----------------------
# API endpoints
# -----------------------
@require_POST
def consent(request):
    _ensure_session_defaults(request)
    request.session["consented"] = True
    request.session.modified = True
    return JsonResponse({"ok": True})

@require_POST
def assign(request):
    _ensure_session_defaults(request)
    if not request.session.get("consented", False):
        return _json_error("Consent required.", 403)

    # random assignment
    if request.session.get("group") not in ("control", "treatment"):
        request.session["group"] = "treatment" if (hash(request.session["participant_id"]) % 2 == 0) else "control"
        request.session.modified = True

    return JsonResponse({"ok": True, "group": request.session["group"]})

@require_POST
def next_item(request):
    _ensure_session_defaults(request)
    if not request.session.get("consented", False):
        return _json_error("Consent required.", 403)
    if MODEL is None:
        return _json_error(f"Model not available: {MODEL_LOAD_ERROR or 'unknown error'}", 500)

    ratings = request.session.get("ratings", {})
    asked = set(request.session.get("asked", []))

    # pick next item (with what-if previews for treatment group)
    payload = MODEL.select_next_item(ratings, asked_set=asked, pool_size=40) or None
    if payload is None:
        return JsonResponse({"ok": True, "done": True})

    # Coerce to JSON-safe types (avoid numpy.int64)
    mid = int(payload["movieId"])
    title = str(payload["title"])

    out = {"ok": True, "movieId": mid, "title": title}

    if "effect" in payload:
        out["effect"] = float(payload["effect"])

    if request.session.get("group") == "treatment":
        wi = payload.get("what_if", {"low": [], "high": []})
        out["what_if"] = {
            "low":  [{"title": str(t[0]), "movieId": int(t[1])} for t in wi.get("low", [])],
            "high": [{"title": str(t[0]), "movieId": int(t[1])} for t in wi.get("high", [])],
        }
    else:
        out["what_if"] = None

    # mark asked immediately (use Python int)
    asked_list = request.session.get("asked", [])
    if mid not in asked_list:
        asked_list.append(mid)
        request.session["asked"] = asked_list
        request.session.modified = True

    return JsonResponse(out)

@require_POST
def submit_rating(request):
    _ensure_session_defaults(request)
    if not request.session.get("consented", False):
        return _json_error("Consent required.", 403)
    if MODEL is None:
        return _json_error(f"Model not available: {MODEL_LOAD_ERROR or 'unknown error'}", 500)

    # ✅ accept JSON body OR form-encoded (both work)
    try:
        data = json.loads(request.body or b"{}")
    except Exception:
        data = {}
    movie_id = data.get("movieId") or request.POST.get("movieId")
    rating   = data.get("rating")  or request.POST.get("rating")

    try:
        movie_id = int(movie_id)
        rating   = float(rating)
    except Exception:
        return _json_error("Invalid parameters.")

    if rating < 0.5 or rating > 5.0:
        return _json_error("Rating must be between 0.5 and 5.0")

    ratings = request.session.get("ratings", {})
    ratings[str(movie_id)] = rating  # keep keys as strings
    request.session["ratings"] = ratings
    request.session.modified = True

    return JsonResponse({"ok": True})

@require_POST
def recommendations(request):
    _ensure_session_defaults(request)
    if not request.session.get("consented", False):
        return _json_error("Consent required.", 403)
    if MODEL is None:
        return _json_error(f"Model not available: {MODEL_LOAD_ERROR or 'unknown error'}", 500)

    ratings = {int(k): float(v) for k, v in request.session.get("ratings", {}).items()}
    u = MODEL.solve_user_vector(ratings)
    exclude = set(ratings.keys())
    top = MODEL.top_n(u, exclude_ids=exclude, n=10)
    out = [{"title": t[1], "movieId": int(t[0]), "score": round(float(t[2]), 3)} for t in top]
    return JsonResponse({"ok": True, "items": out})

@require_POST
def survey_submit(request):
    _ensure_session_defaults(request)

    if not request.session.get("consented", False):
        return _json_error("Consent required.", 403)

    # ✅ accept JSON body OR form-encoded
    try:
        data = json.loads(request.body or b"{}")
    except Exception:
        data = {}
    feedback = (data.get("feedback") or request.POST.get("feedback") or "").strip()

    group = request.session.get("group")
    pid = request.session.get("participant_id")
    ts = datetime.datetime.utcnow().isoformat()

    # Persist to CSV (course-project simple)
    base_dir = getattr(settings, "BASE_DIR", os.getcwd())
    out_path = os.path.join(base_dir, "project4_survey.csv")
    try:
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["timestamp_utc", "participant_id", "group", "feedback"])
            w.writerow([ts, pid, group, feedback])
    except Exception as e:
        return _json_error(f"Failed to save feedback: {e}", 500)

    return JsonResponse({"ok": True, "msg": "Thanks for your feedback!"})
