# Django Libraries
from django.shortcuts import render
from django.http import HttpResponse , JsonResponse

# Rest framework import
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes,authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication , TokenAuthentication

# import other class
# from .excel_processing import *
from .msd_processing import *


def msd(request):
    final_json = {}
    # getting value from query string
    file_name_list = request.GET.getlist('file','no file')
    print('file_list',file_name_list)
    if file_name_list == 'no file':
        return render(request, 'extractor/index_msd.html')
        # return Response({'status':'0'})
    else:
        pass
    for file_index , file_name in enumerate(file_name_list):
        doc_format = os.path.splitext(file_name)[1].lower()
        if doc_format == '.docx':
            output = msd_extraction().main(file_name)
            final_json[file_index] = output
        else:
            final_json[file_index] = {'status':0}
    return JsonResponse(final_json)

def dataset_to_mangodb(request):
    from pymongo import MongoClient
    client = MongoClient('172.28.42.150',27017)
    db = client['dataset']
    collection = db['msd_content']
    data = [msd_content(category=i['category'], text=i['text'], language_code=i['language_code'],
                        language=i['language'],
                        category_actual=i['category_actual'],
                        type=i['type']) for i in collection.find({})]
    if data:
        msd_content.objects.bulk_create(data)
        return HttpResponse('success')
    else:
        return HttpResponse('Failure')