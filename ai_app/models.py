from django.db import models
import os


def path_and_rename1(instance, filename):
    upload_to = 'audio/pump1'
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format('pump1', ext)
    return os.path.join(upload_to, filename)


def path_and_rename2(instance, filename):
    upload_to = 'audio/pump2'
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format('pump2', ext)
    return os.path.join(upload_to, filename)


def path_and_rename3(instance, filename):
    upload_to = 'audio/pump2'
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format('pump2', ext)
    return os.path.join(upload_to, filename)


class Classifier(models.Model):
    pump1_file = models.FileField(
        upload_to=path_and_rename1, default='audio/None/no-img.jpg')

    pump2_file = models.FileField(
        upload_to=path_and_rename2, default='audio/None/no-img.jpg')

    pump3_file = models.FileField(
        upload_to=path_and_rename3, default='audio/None/no-img.jpg')

    category = models.CharField(max_length=20)
