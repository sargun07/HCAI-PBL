from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path('index', views.index, name='index'),
    path('save-layout/', views.save_layout, name='save_layout'),
    path("run-episode/", views.run_episode, name="run_episode"),
    path("sample-pair/", views.sample_pair, name="sample_pair"),
    path("submit-pref/", views.submit_pref, name="submit_pref"),
    path("train-reward-now/", views.train_reward_now, name="train_reward_now"),
    path("train-rlhf-now/", views.train_rlhf_now, name="train_rlhf_now"),
    path("train-reinforce-now/", views.train_reinforce_now, name="train_reinforce_now"),
    path("reset-training/", views.reset_training, name="reset_training"),
    path("clear-preferences/", views.clear_preferences, name="clear_preferences"),
    path("download-model/<str:kind>/", views.download_model, name="download_model"),
    path("download-models-zip/", views.download_models_zip, name="download_models_zip"),

]
