from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path('index', views.index, name='index'),  
]
