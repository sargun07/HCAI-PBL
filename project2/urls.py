# urls.py
from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path('index', views.index, name='index'),
    path('run-active-learning', views.run_active_learning, name='run_al'),
    path('train', views.train, name='train'),
    path('download-model', views.download_model, name='download_model'),
    path('download-dataset', views.download_dataset, name='download_dataset'),
    path('al/human/init', views.al_human_init, name='al_human_init'),
    path('al/human/next', views.al_human_next, name='al_human_next'),
    path('al/human/label', views.al_human_label, name='al_human_label'),
    path('al/human/reset', views.al_human_reset, name='al_human_reset'),

]
