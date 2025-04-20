import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt


from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse

def index(request):
    return HttpResponse("Welcome to Project 1!")


def upload_csv(request):
    result = None
    error = None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            decoded_file = file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)

            try:
                reader = csv.reader(io_string)
                numbers = []

                for row in reader:
                    for item in row:
                        try:
                            numbers.append(float(item.strip()))
                        except ValueError:
                            pass  # Skip non-numeric values

                if numbers:
                    result = sum(numbers) / len(numbers)
                else:
                    error = "No numeric values found in the CSV."
            except Exception as e:
                error = f"Error processing file: {str(e)}"
    else:
        form = CSVUploadForm()

    return render(request, 'demos/upload.html', {
        'form': form,
        'result': result,
        'error': error
    })



def generate_plot(request):
    filename = 'myplot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.scatter(x, y)
    plt.savefig(image_path)
    
    image_url = settings.MEDIA_URL + filename
    return render(request, 'demos/show_plot.html', {'image_url': image_url})