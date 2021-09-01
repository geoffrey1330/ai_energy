from rest_framework import serializers

from .models import Classifier


class PumpSerializer(serializers.ModelSerializer):
    """
    pump Serializer.
    """
    class Meta:
        model = Classifier
        fields = "__all__"
