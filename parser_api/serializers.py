from rest_framework.serializers import Serializer
from rest_framework import serializers


class ParserSerializer(Serializer):
    summary = serializers.CharField(required=True)
    experience = serializers.CharField(required=True)
    education = serializers.CharField(required=True)
    certification = serializers.CharField(required=True)
    requests = serializers.ListField(required=True)
