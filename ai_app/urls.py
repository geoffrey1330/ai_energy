from django.urls import path

from . import views

urlpatterns = [
    path('pump1/', views.pump1),
    path('pump2/', views.pump2),
    path('pump3/', views.pump3),

]
