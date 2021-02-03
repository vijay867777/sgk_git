from django.shortcuts import render
from . import views
from django.urls import path, include , re_path
from rest_framework.authtoken.views import obtain_auth_token
from django.views.generic.base import RedirectView
from django.contrib import admin
from django.views.generic import TemplateView
from django.conf.urls import include, url
from django.views.static import serve


urlpatterns = [
    re_path(r"msd\/$", views.msd, name='msd'),
    # re_path(r"classifier\/$", views.classifier, name='classifier'),
    # path("extractor/", views.extractor, name='extractor'),
    # re_path(r"doc_extractor\/$",views.doc_extractor,name='doc_extractor'),
    re_path(r"api_token\/$",obtain_auth_token,name='auth_token'),
    re_path(r"dataset_update\/$",views.dataset_to_mangodb,name='dataset_update'),
    ]