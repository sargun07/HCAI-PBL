from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path('index', views.index, name='index'),
]