# project4/urls.py
from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("index/", views.landing, name="index"),          
    path("study/", views.index, name="study"),          
    path("api/consent", views.consent, name="consent"),
    path("api/assign", views.assign, name="assign"),
    path("api/next_item", views.next_item, name="next_item"),
    path("api/submit_rating", views.submit_rating, name="submit_rating"),
    path("api/recommendations", views.recommendations, name="recommendations"),
    path("api/survey_submit", views.survey_submit, name="survey_submit"),
]
