from django.urls import path, include
from . import views

app_name = 'home'

urlpatterns = [
    path("", views.index, name="index"),
    path('project1/', include('project1.urls'))
]