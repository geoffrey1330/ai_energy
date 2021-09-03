from rest_framework import status as s
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import login, logout, authenticate

# Create your views here.
from .serializers import UserSerializer, LoginSerializer
from .models import User


@api_view(['POST'])
def login_user(request: Request) -> Response:
    
    if request.user.is_authenticated:
        return Response(status=s.HTTP_400_BAD_REQUEST)

    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        user = authenticate(request, username=serializer.validated_data['username'],
                            password=serializer.validated_data['password'])
        
        if user:
            login(request, user)
            return Response(UserSerializer(instance=user).data)

        return Response({'err': 'Invalid credentials'}, status=s.HTTP_403_FORBIDDEN)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_user(request: Request) -> Response:
    logout(request)
    return Response()
