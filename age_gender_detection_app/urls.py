# age_gender_detection_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_result, name='upload_result'),
    path('delete_image/', views.delete_image, name='delete_image'),
]
