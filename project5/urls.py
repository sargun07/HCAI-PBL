from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path('index', views.index, name='index'),
]