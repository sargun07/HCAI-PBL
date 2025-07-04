import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json

from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render
from .utils import DataInvestigator, TargetColumnHandler
from .forms import CSVUploadForm
from django.http import HttpResponse, JsonResponse
from .ml_util import get_model_instance

def index(request):

    if request.method == 'POST':

        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():

            file = request.FILES['file']
            target_column = request.POST.get('target_column')
            decoded_file = file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            try:

                # identifying the dependent column
                df = pd.read_csv(io_string)
                columnHandler = TargetColumnHandler(df, target_column)
                df = columnHandler.get_processed_df()

                # investigating the data
                investigator = DataInvestigator(df)  
                result = investigator.perform_analysis()

                model_recommendation = result["Model Recommendation"]
                dataset_summary = result["Dataset Summary"][1]
                plot  = result["Data Analyser"]

                # storing the dataframe as a csv file
                save_dir = os.path.join(settings.MEDIA_ROOT, "processed")
                save_path = os.path.join(save_dir, "project1_uploaded_data.csv")

                os.makedirs(save_dir, exist_ok=True)

                if os.path.exists(save_path):
                    os.remove(save_path)

                df.to_csv(save_path, index=False)


                return JsonResponse({
                    'success': True,
                    'recommendation': result["Model Recommendation"],
                    'summary': result["Dataset Summary"][1],
                    'plots': result["Data Analyser"]
                })


            except Exception as e:
                return JsonResponse({'success': False, 'error': f"Error processing file: {str(e)}"})
        else:
            return JsonResponse({'success': False, 'error': "Invalid form submission."})
    else:
        form = CSVUploadForm()

    return render(request, 'project1/interface.html', {'form': form})


@csrf_exempt
def train_model_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        ml_type = data.get("ml_type")
        model_name = data.get("model")
        hyperparameters = data.get("hyperparameters")
        train_split = data.get("train_split")
        scoring_metric = data.get("scoring_metric")

        try:
            # Load the stored dataset
            df_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_uploaded_data.csv")
            df = pd.read_csv(df_path)
            X = df.drop(columns=[df.columns[-1]])
            y = df[df.columns[-1]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_split / 100), random_state=42)

            model = get_model_instance(model_name, hyperparameters)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # Save model
            model_path = os.path.join(settings.MEDIA_ROOT, "processed", "project1_trained_model.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            joblib.dump(pipeline, model_path)
            
            # Generate plot if classification
            plot_base64 = None
            if ml_type == "classification":
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots()
                disp.plot(ax=ax)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            elif ml_type == "regression":
                y_pred = pipeline.predict(X_test)
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")

                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)

            return JsonResponse({
                "success": True,
                "score": round(score, 4),
                "plot": plot_base64
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
