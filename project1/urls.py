from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path('index', views.index, name='index'),  
    path('train_model/', views.train_model_view, name='train_model'),
]
