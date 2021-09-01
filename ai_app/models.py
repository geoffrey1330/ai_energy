from django.db import models


class Classifier(models.Model):
    category = models.CharField(max_length=20)
