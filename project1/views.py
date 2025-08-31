import io, os, json, base64
import pandas as pd
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .utils import DataInvestigator, TargetColumnHandler
from .forms import CSVUploadForm
from .ml_util import get_model_instance

# NEW: sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib
from matplotlib import pyplot as plt

def index(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            target_column = request.POST.get('target_column')
            decoded_file = file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            try:
                # Load & place target at end
                df = pd.read_csv(io_string)
                columnHandler = TargetColumnHandler(df, target_column)
                df = columnHandler.get_processed_df()

                # Investigate
                investigator = DataInvestigator(df)
                result = investigator.perform_analysis()

                # Persist dataset for training/prediction
                save_dir = os.path.join(settings.MEDIA_ROOT, "processed")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "project1_uploaded_data.csv")
                if os.path.exists(save_path):
                    os.remove(save_path)
                df.to_csv(save_path, index=False)

                return JsonResponse({
                    "success": True,
                    "recommendation": result["Model Recommendation"],
                    "summary": result["Dataset Summary"][1],
                    "plots": result["Data Analyser"]
                })
            except Exception as e:
                return JsonResponse({"success": False, "error": f"Error processing file: {str(e)}"})
        else:
            return JsonResponse({"success": False, "error": "Invalid form submission."})
    else:
        form = CSVUploadForm()
    return render(request, "project1/interface.html", {"form": form})

@csrf_exempt
def train_model_view(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required."})

    data = json.loads(request.body)
    ml_type = data.get("ml_type")
    model_name = data.get("model")
    hyperparameters = data.get("hyperparameters") or {}
    train_split = float(data.get("train_split", 80))
    scoring_metric = data.get("scoring_metric")

    try:
        # Load the stored dataset
        df_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_uploaded_data.csv")
        if not os.path.exists(df_path):
            return JsonResponse({"success": False, "error": "No uploaded dataset found. Upload a CSV first."})

        df = pd.read_csv(df_path)
        X = df.drop(columns=[df.columns[-1]]).copy()
        y = df[df.columns[-1]].copy()

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

        preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],remainder="drop",)

        model = get_model_instance(model_name, hyperparameters)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_split / 100.0), random_state=42, stratify=y if ml_type == "classification" else None
        )

        # Fit
        pipeline.fit(X_train, y_train)

        # Score
        if ml_type == "classification":
            y_pred = pipeline.predict(X_test)
            if scoring_metric == "f1":
                score = f1_score(y_test, y_pred, average="weighted")
            elif scoring_metric == "precision":
                score = precision_score(y_test, y_pred, average="weighted")
            elif scoring_metric == "recall":
                score = recall_score(y_test, y_pred, average="weighted")
            else:
                score = accuracy_score(y_test, y_pred)

            # Confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
            fig, ax = plt.subplots()
            disp.plot(ax=ax , cmap="viridis", values_format="d", colorbar=True)
        else:
            y_pred = pipeline.predict(X_test)
            if scoring_metric == "neg_mean_squared_error":
                score = -mean_squared_error(y_test, y_pred)
            elif scoring_metric == "neg_mean_absolute_error":
                score = -mean_absolute_error(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)

            # Regression scatter plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
            y_min, y_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # Save trained pipeline
        model_dir = os.path.join(settings.MEDIA_ROOT, "processed")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "project1_trained_model.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        joblib.dump(pipeline, model_path)

        return JsonResponse({
            "success": True,
            "score": round(float(score), 4),
            "plot": plot_base64,
            "model_path": "/media/processed/project1_trained_model.pkl"
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})

# @csrf_exempt
# def predict_view(request):
#     if request.method != "POST":
#         return JsonResponse({"success": False, "error": "POST required."})
#     try:
#         data = json.loads(request.body)
#         raw = data.get("values", "")
#         # parse "4.5, 2.3, 1.4, 0.2"
#         values = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
#         model_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_trained_model.pkl")
#         if not os.path.exists(model_path):
#             return JsonResponse({"success": False, "error": "No trained model found. Train a model first."})
#         pipe = joblib.load(model_path)
#         import numpy as np
#         arr = np.array(values, dtype=float).reshape(1, -1)
#         pred = pipe.predict(arr)
#         return JsonResponse({"success": True, "prediction": str(pred[0])})
#     except Exception as e:
#         return JsonResponse({"success": False, "error": str(e)})

@csrf_exempt
def predict_view(request):
    """
    Predict for a single input row provided as a comma-separated string.

    - Uses training schema (feature names + dtypes) from the processed CSV.
    - Loads the saved pipeline (preprocessor + model).
    - VALIDATES categorical inputs against the OneHotEncoder's learned categories.
      If a value is not in the training categories, returns a 422 with suggestions.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required."}, status=405)

    try:
        import os, json, re
        import pandas as pd
        import joblib
        from django.conf import settings
        from pandas.api import types as pdt

        # Toggle this if you prefer to warn instead of error on unknown categories
        STRICT_CATEGORICALS = True

        data = json.loads(request.body or "{}")
        raw = str(data.get("values", "")).strip()
        if not raw:
            return JsonResponse({"success": False, "error": "Missing 'values' in request body."}, status=400)

        # Recover schema from the processed CSV used for training
        csv_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_uploaded_data.csv")
        if not os.path.exists(csv_path):
            return JsonResponse({"success": False, "error": "No uploaded dataset found. Upload and train first."}, status=400)

        df_schema = pd.read_csv(csv_path)
        if df_schema.shape[1] < 2:
            return JsonResponse({"success": False, "error": "Processed CSV seems invalid (need at least 1 feature + target)."}, status=400)

        feature_cols = df_schema.columns[:-1]  # target assumed last
        X_schema = df_schema[feature_cols]

        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != len(feature_cols):
            return JsonResponse(
                {"success": False,
                 "error": f"Expected {len(feature_cols)} values (features only, no target). "
                          f"Got {len(parts)}.\nExpected order: {', '.join(feature_cols)}"},
                status=400
            )

        # Build typed row (numerics -> float, categoricals -> str)
        row = {}
        for col, val in zip(feature_cols, parts):
            dtype = X_schema[col].dtype
            if pdt.is_numeric_dtype(dtype):
                if isinstance(val, str) and val.lower() in ("true", "false"):
                    row[col] = 1.0 if val.lower() == "true" else 0.0
                else:
                    try:
                        row[col] = float(val)
                    except Exception:
                        return JsonResponse(
                            {"success": False, "error": f"Column '{col}' expects numeric, got '{val}'."},
                            status=400
                        )
            else:
                row[col] = val  # keep strings for categoricals

        X_one = pd.DataFrame([row], columns=feature_cols)

        # Load pipeline (preprocessor + model)
        model_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_trained_model.pkl")
        if not os.path.exists(model_path):
            return JsonResponse({"success": False, "error": "No trained model found. Train a model first."}, status=400)
        pipe = joblib.load(model_path)

        # ---- Categorical validation against the fitted OneHotEncoder ----
        # Find the OneHotEncoder and its column list inside the ColumnTransformer
        pre = getattr(pipe, "named_steps", {}).get("preprocessor", None)
        if pre is not None and hasattr(pre, "transformers_"):
            cat_transformer = None
            cat_cols = None

            for name, transformer, cols in pre.transformers_:
                # If you named it "cat" when building the preprocessor, this will match quickly
                if name == "cat":
                    cat_transformer = transformer
                    cat_cols = list(cols) if hasattr(cols, "__iter__") else [cols]
                    break

            # Handle case where the categorical branch is itself a Pipeline
            if hasattr(cat_transformer, "named_steps"):
                # Try common step names
                if "onehot" in cat_transformer.named_steps:
                    ohe = cat_transformer.named_steps["onehot"]
                else:
                    # Fallback: first OneHotEncoder-like step
                    ohe = next((s for s in cat_transformer.named_steps.values()
                                if s.__class__.__name__.lower().startswith("onehotencoder")), None)
            else:
                ohe = cat_transformer

            # If we found an OHE and the column list, validate values
            if ohe is not None and cat_cols is not None and hasattr(ohe, "categories_"):
                unknowns = []
                # Map: column name -> allowed categories
                for idx, col in enumerate(cat_cols):
                    allowed = set(map(str, ohe.categories_[idx]))
                    val = str(X_one.at[0, col])

                    # If numeric-looking provided for a categorical, it will still be a string here;
                    # we simply check membership in allowed categories.
                    if val not in allowed:
                        unknowns.append((col, val, sorted(allowed)))

                if unknowns and STRICT_CATEGORICALS:
                    # Build a helpful message listing the first few valid options per offending column
                    msgs = []
                    for col, val, allowed_list in unknowns:
                        preview = ", ".join(allowed_list[:10]) + ("..." if len(allowed_list) > 10 else "")
                        msgs.append(f"Column '{col}': value '{val}' not in training categories. Valid options include: {preview}")
                    return JsonResponse(
                        {"success": False, "error": "Invalid categorical value(s). " + " | ".join(msgs)},
                        status=422
                    )
                elif unknowns and not STRICT_CATEGORICALS:
                    # Optionally attach a warning and proceed (OHE with handle_unknown='ignore' will zero-encode)
                    pass

        # ---- Predict ----
        pred = pipe.predict(X_one)
        return JsonResponse({"success": True, "prediction": str(pred[0])})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
