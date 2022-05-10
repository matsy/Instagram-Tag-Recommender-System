from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.

from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, viewsets
from .models import File

from .serializers import FileSerializer


class FileUploadView(viewsets.ModelViewSet):
  queryset = File.objects.all()
  serializer_class = FileSerializer

  def post(self, request, *args, **kwargs):
    caption = request.data['caption']
    File.objects.create(caption=caption)
    return HttpResponse({'message: caption image uploaded'}, status=200)

  # parser_class = (FileUploadParser,)
  #
  # def post(self, request, *args, **kwargs):
  #
  #   file_serializer = FileSerializer(data=request.data)
  #
  #   if file_serializer.is_valid():
  #     file_serializer.save()
  #     return Response(file_serializer.data, status=status.HTTP_201_CREATED)
  #   else:
  #     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
