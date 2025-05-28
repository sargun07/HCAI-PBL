from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path('index', views.index, name='index'),
]