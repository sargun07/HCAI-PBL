import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


from django.conf import settings
from django.shortcuts import render
from .utils import DataInvestigator, TargetColumnHandler
from .forms import CSVUploadForm
from django.http import HttpResponse, JsonResponse

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
