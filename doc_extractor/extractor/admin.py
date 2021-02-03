from django.contrib import admin
from .models import *

# Register your models here.

admin.site.site_title = "SGK API INTERFACE"
admin.site.site_header = "SGK API ADMIN"

# admin.site.register(msd_dataset)

class view_msd_dataset(admin.ModelAdmin):
    list_display = ('text','category','language_code')
    # list_display = ('text','category','language_code','language','type')
    list_per_page = 30
    # search_fields = ['category',]
    list_filter = ('language_code','category',)

    def text(self, object):
        return object.text

    def category(self, object):
        return object.category

    def language_code(self, object):
        return object.language_code

    # def language(self, object):
    #     return object.language
    #
    # def type(self, object):
    #     return object.type

admin.site.register(msd_dataset,view_msd_dataset)

class view_msd_dataset_contents(admin.ModelAdmin):
    list_display = ('text','category','language_code')
    # list_display = ('text','category','language_code','language','type')
    list_per_page = 30
    # search_fields = ['category',]
    list_filter = ('language_code','category',)

    def text(self, object):
        return object.text

    def category(self, object):
        return object.category

    def language_code(self, object):
        return object.language_code

    # def language(self, object):
    #     return object.language
    #
    # def type(self, object):
    #     return object.type

admin.site.register(msd_content,view_msd_dataset_contents)

class undetected_msd_logbook(admin.ModelAdmin):
    list_display = ('text','header_category','content_category','language_code')
    # list_display = ('text','category','language_code','language','type')
    list_per_page = 30
    # search_fields = ['category',]
    list_filter = ('language_code','category',)

    def text(self, object):
        return object.text

    def header_category(self, object):
        return object.header_category

    def content_category(self, object):
        return object.content_category

    def language_code(self, object):
        return object.language_code

admin.site.register(undetected_msd_log,undetected_msd_logbook)