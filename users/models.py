from django.db import models
from django.contrib.auth.models import AbstractUser
from uuid import uuid4


class User(AbstractUser):
    """
    ai_energy user model
    """
    date_joined = models.DateField(auto_now_add=True)
