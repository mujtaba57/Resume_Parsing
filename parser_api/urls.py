from django.urls import path, include
from .views import ParserView


urlpatterns = [
    path('parser/',  ParserView.as_view()),
]
