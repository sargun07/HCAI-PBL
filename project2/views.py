from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from .pipeline import IMDBDataLoader, SVMTextClassifier
import pandas as pd
import os

def index(request):
    try: 
        filepath = os.path.join(settings.MEDIA_ROOT, 'project2')
        dataLoader = IMDBDataLoader(filepath, 'IMDB Dataset.csv')
        
        df = dataLoader.load_data()

        #  data summary:
        sample = df.sample(10).to_dict(orient='records')
        total_reviews = len(df)
        num_pos = df[df['sentiment'] == 'positive'].shape[0]
        num_neg = df[df['sentiment'] == 'negative'].shape[0]

        # data preprocessing:
        # dataLoader.preprocess()

        # saving the preprocessed file
        # dataLoader.save_preprocessed_data()

        # output_filepath = os.path.join(os.path.dirname(filepath), "imdb_dataset_preprocessed.csv")
        # preprocessed_df = pd.read_csv(output_filepath)

        # classifier = SVMTextClassifier()
        # classifier.train(preprocessed_df)
        model_filepath = os.path.join(filepath, 'svm_model.pkl')
        # results = classifier.evaluate()
        # classifier.save_model(model_filepath)
    
        clf = SVMTextClassifier.load(model_filepath)
        results = {
            "accuracy": clf.accuracy,
            "confusion_matrix": clf.confusion_matrix_,
            "report": clf.report
        }

    except Exception as e:
        return JsonResponse({'success': False, 'error': f"Error processing file: {str(e)}"})
            
    return render(request, 'project2/interface.html', {
        'sample_reviews': sample,
        'total_reviews': total_reviews,
        'num_pos': num_pos,
        'num_neg': num_neg,
        "accuracy": results["accuracy"],
        "report": results["report"],
        "confusion_matrix": results["confusion_matrix"]
    })