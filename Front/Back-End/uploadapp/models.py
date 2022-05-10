from django.db import models


# Create your models here.
# from .models import File

def upload_path(instance, filename):
  return '/'.join(['caption', filename])


class File(models.Model):
  caption = models.ImageField(blank=False, null=False, upload_to=upload_path)
