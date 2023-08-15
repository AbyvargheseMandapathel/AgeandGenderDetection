# age_gender_detection_app/models.py

from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    age_predictions = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.image.name
