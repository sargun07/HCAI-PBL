from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", include("home.urls")),
    path("admin/", admin.site.urls),
    path("demos/", include("demos.urls")),
    path("project1/", include("project1.urls")),
    path("project2/", include("project2.urls")),
    path("project3/", include("project3.urls")),
    path("project4/", include("project4.urls")),
    path("project5/", include("project5.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)