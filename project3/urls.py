from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path('index', views.index, name='index'),
    path("train/", views.train_api, name="train_api"),
    path("counterfactuals/", views.counterfactuals_api, name="counterfactuals_api"),
]