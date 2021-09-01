import scipy.stats
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import sys
import os
import requests
import librosa
#from django.shortcuts import render
#from django.http import HttpResponse, JsonResponse
# Create your views here.
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Classifier
from .serializers import PumpSerializer
from django.conf import settings

model_1 = './models/model_pump_section_00.hdf5'
model_2 = './models/model_pump_section_01.hdf5'
model_3 = './models/model_pump_section_02.hdf5'

# import boto3

# s3 = boto3.client('s3')
# s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')

#import libraries


# Response from request

section_idx = 0

file_path = "./models/section_00_source_test_anomaly_0000.wav"

# constants
SAMPLE_RATE = 22050
decision_threshold = 0.9
#model_dir = f"Models/MLP_Autoencoder/{machine_type}"

n_mels = 128
n_frames = 64
n_hop_frames = 8
n_fft = 10
hop_length = 512
power = 2.0


def file_to_vectors(file_path,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa

    signal, sr = librosa.load(file_path, sr=None, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * \
        np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1
    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t: n_mels *
                (t + 1)] = log_mel_spectrogram[:, t: t + n_vectors].T

    return vectors


def predictor(data, model):
    p = model.predict(data)

    y_pred = np.mean(np.square(data - p))
    if y_pred > decision_threshold:
        print("it's 1")
        decision_result = 1  # [os.path.basename(file_path), 1]
    else:
        print("it's 0")
        decision_result = 0
    return decision_result


@api_view(['GET', 'POST'])
def pump1(request):

    # if request.method == 'GET':
    #     transformer = Classifier.objects.all()
    #     serializer = PumpSerializer(transformer, many=True)
    #     return JsonResponse(serializer.data, safe=False)

    # extract data
    data = file_to_vectors(file_path,
                           n_mels=n_mels,
                           n_frames=n_frames,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           power=power)

    model = load_model(model_1)

# make prediction

    decision_result = predictor(data, model)

    if request.method == 'GET':
       #data = JSONParser().parse(request)
        data = {"category": decision_result}
        serializer = PumpSerializer(data=data)
        if serializer.is_valid():
            # serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'POST'])
def pump2(request):

    # if request.method == 'GET':
    #     transformer = Classifier.objects.all()
    #     serializer = PumpSerializer(transformer, many=True)
    #     return JsonResponse(serializer.data, safe=False)

    # extract data
    data = file_to_vectors(file_path,
                           n_mels=n_mels,
                           n_frames=n_frames,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           power=power)

    model = load_model(model_2)

# make prediction

    decision_result = predictor(data, model)

    if request.method == 'GET':
       #data = JSONParser().parse(request)
        data = {"category": decision_result}
        serializer = PumpSerializer(data=data)
        if serializer.is_valid():
            # serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'POST'])
def pump3(request):

    data = file_to_vectors(file_path,
                           n_mels=n_mels,
                           n_frames=n_frames,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           power=power)

    model = load_model(model_3)

# make prediction

    decision_result = predictor(data, model)

    if request.method == 'GET':
       #data = JSONParser().parse(request)
        data = {"category": decision_result}
        serializer = PumpSerializer(data=data)
        if serializer.is_valid():
            # serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
