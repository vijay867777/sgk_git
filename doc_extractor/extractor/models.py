# from django.db import models
from djongo import models

# Create your models here.
class msd_dataset(models.Model):
    category = models.CharField(max_length=50,null=False)
    text = models.TextField(null=False)
    language_code = models.CharField(max_length=20,blank=True)
    language = models.CharField(max_length=20,blank=True)
    type = models.CharField(max_length=50,blank=True)

class msd_content(models.Model):
    text = models.TextField(null=False)
    category_actual = models.CharField(max_length=50,null=False)
    category = models.CharField(max_length=50,null=False)
    language_code = models.CharField(max_length=20,blank=True)
    language = models.CharField(max_length=20,blank=True)
    type = models.CharField(max_length=50, blank=True)

class undetected_msd_log(models.Model):
    file_name = models.TextField(default='')
    text = models.TextField(null=False)
    header_category = models.CharField(max_length=50,null=False)
    content_category = models.CharField(max_length=50,null=False)
    language_code = models.CharField(max_length=20,blank =True,default='')
    category = models.CharField(max_length=50,default='')
    language = models.CharField(max_length=20,blank=True,default='')

