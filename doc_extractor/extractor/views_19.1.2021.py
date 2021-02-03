# Django Libraries
# import concurrent.futures
from django.shortcuts import render
from django.http import HttpResponse , JsonResponse
from .models import *

# Custom Libraries
import pandas as pd
import numpy as np
import joblib
from laserembeddings import Laser
# from sentence_transformers import SentenceTransformer , models
from sklearn.neural_network import MLPClassifier                            # works great -- neural network
from langid import classify
from langdetect import detect
# from fastlangid.langid import LID
import os
import re
import pdfplumber
from docx2json import convert
import json
from docx import Document
import smbclient
import mammoth
from bs4 import BeautifulSoup

# Rest framework import
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes,authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication , TokenAuthentication

# Other file import
from environment import MODE

if MODE == 'local':
    from .local_constants import *
else:
    from .dev_constants import *

from .excel_processing import *
from .msd_processing import *

categories = ['nutrition','ingredients','allergen statement','shelf_life_statement',
              'storage instruction','address',
              # 'warning statement',
              "gtin_number","serial_number","lot_number","expiry_date",'form_content',
              'usage instruction','pc_number','general classification',"eu_number"]

msd_categories = ['name','active_substance','excipients','form_content','method_route','warning','expiry_date',
                  'storage_instructions','precautions','marketing_company','unique_identifier','classification',
                  'usage_instruction','braille_info','mfg_date','manufacturer','packing_site','appearance',
                  'product_info','label_dosage','box_info']

# Initialize Laser
# model_path = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/labse"
# model =  SentenceTransformer(model_path)

# Initialize Laser
laser = Laser(path_to_bpe_codes,path_to_bpe_vocab,path_to_encoder)

# langid = LID()

# @authentication_classes([SessionAuthentication, BasicAuthentication])
# @api_view()
# @permission_classes([IsAuthenticated])
# @authentication_classes([TokenAuthentication])
# def extractor(request):
#     content = {'message': 'Hello, World!'}
#     return Response(content)
    # return render(request,'extractor/index.html')

# Create your views here.
def model_training():
    df = pd.read_excel(input_excel)
    df = df.sample(frac=1)
    X_train_laser = laser.embed_sentences(df['text'], lang='en')
    # X_train_laser = model.encode(df['text'])
    # mlp = MLPClassifier(hidden_layer_sizes=(125,), solver='adam', activation='tanh', random_state=0, shuffle=True)
    mlp = MLPClassifier(hidden_layer_sizes=(70,),solver='adam',max_iter=500,activation='tanh',random_state=0,shuffle=True)
    mlp.fit(X_train_laser, df['category'])
    joblib.dump(mlp,model_location)
    return mlp

def classifier(request):
    text = request.GET.get('text','')
    if text:
        pass
    else:
        return render(request, 'extractor/index_classifier.html')
    model = None
    if os.path.exists(model_location):
        model = joblib.load(model_location)
    else:
        model = model_training()
    # lang_detected = detect(text)
    lang_detected = classify(text)[0]
    # print('lang----->',lang_detected)
    # print(text)
    prediction = model.predict(laser.embed_sentences([text],lang=lang_detected))
    probability = model.predict_proba(laser.embed_sentences([text],lang=lang_detected))
    probability[0].sort()
    max_probability = max(probability[0])
    if (max_probability-0.35) > probability[0][-2]:
        pred_output = prediction[0]
    else:
        pred_output = 'None'
    # print(probability)
    print('{}-------------->{}'.format(max(probability[0]),pred_output))
    result = {'probability':max(probability[0]),'output':pred_output,'actual_output':prediction[0],'text':text}
    # return HttpResponse(pred_output)
    # return render(request,'extractor/doc_result.html',{'result':dict})
    return render(request,'extractor/index_result.html',result)

def prediction(text):
    if os.path.exists(model_location):
        model = joblib.load(model_location)
    else:
        model = model_training()
    # lang_detected = detect(text)
    lang_detected = classify(text)[0]
    print(text)
    prediction = model.predict(laser.embed_sentences([text],lang=lang_detected))
    # prediction = model.predict(model.encode([text]))
    probability = model.predict_proba(laser.embed_sentences([text],lang=lang_detected))
    # probability = model.predict_proba(model.encode['text'])
    probability[0].sort()
    max_probability = max(probability[0])
    # if (max_probability-0.35) > probability[0][-2]:
    if max_probability > 0.63:
        pred_output = prediction[0]
    else:
        pred_output = 'None'
    print('{}-------------->{}'.format(max(probability[0]),pred_output))
    return ({'probability': max(probability[0]), 'output': pred_output, 'actual_output': prediction[0]})

def doc_extractor(request):
    final = {}
    file_name = request.GET.get('file','no file')
    if file_name == 'no file':
        return render(request, 'extractor/index.html')
    else:
        pass
    file = document_location+file_name
    doc_format = os.path.splitext(file_name)[1].lower()
    if doc_format == ".pdf":
        if os.path.exists(file):
            pdf = pdfplumber.open(file)
        else:
            return HttpResponse('File not found')
        no_of_pages = len(pdf.pages)
        tables = len(pdf.pages[0].extract_tables())
        if tables > 2:
            print('type 1 --- tables')
            for page_no in range(no_of_pages):
                page = pdf.pages[page_no]
                extracted_table = page.extract_tables()
                text = [" ".join(list(filter(None, content))).replace('\n', ' ') for table in extracted_table for content in table]
                for sentence in text:
                    unique_identifiers = Regex_parsers(sentence)
                    if unique_identifiers:
                        final = {**final, **unique_identifiers}
                    else:
                        pass
                    result = prediction(sentence)['output']
                    if result != 'None':
                        if result in final.keys():
                            final[result].append(sentence)
                        else:
                            final[result] = [sentence]
                    else:
                        pass
            if len(final['Nutrition']) > 1:
                final['Nutrition'] = final['Nutrition'][:-1]
            else:
                pass
            extracted_categories = {key:val for key, val in final.items() if key.lower() in categories}
            # return JsonResponse(extracted_categories)
            return render(request, 'extractor/doc_result.html', {'result': extracted_categories})
        else:
            print('type-2-paragraph')
            for page_no in range(no_of_pages):
                page = pdf.pages[page_no]
                extracted_text = page.extract_text()
                text = sentence_tokennizer(extracted_text)
                for sentence in text:
                    unique_identifiers = Regex_parsers(sentence)
                    if unique_identifiers:
                        final = {**final, **unique_identifiers}
                    else:
                        pass
                    result = prediction(sentence)['output']
                    if result in final.keys():
                        final[result].append(sentence)
                    else:
                        final[result] = [sentence]
            extracted_categories = {key:val for key, val in final.items() if key.lower() in categories}
            return render(request, 'extractor/doc_result.html', {'result': extracted_categories})
    elif (doc_format == '.docx') or (doc_format == '.doc'):
        doc = convert(file,sepBold=True)
        doc_to_json = json.loads(doc)
        text = doc_to_json['nonbold']
        if text:
            pass
        else:
            text = doc_to_json['text']
        for sentence in text:
            unique_identifiers = Regex_parsers(sentence,regex_patterns)
            if unique_identifiers:
                final = {**final,**unique_identifiers}
            else:
                pass
            result = prediction(sentence)['output']
            if result in final.keys():
                final[result].append(sentence)
            else:
                final[result] = [sentence]
        # print(final)
        extracted_categories = {key: val for key, val in final.items() if key.lower() in categories}
        return render(request, 'extractor/doc_result.html', {'result': extracted_categories})
    else:
        return HttpResponse('This file format not supported currently')

def sentence_tokennizer(text):
    #sentences = re.split(r"[.!?]", text)
    # sentences = re.split(r"\.\s\n", text)
    segments = re.split(r"\n\s\n", text)
    sentences = [re.split(r"\.\s\n", seg) for seg in segments]
    # token = [re.sub(r"\d\-.*",'number',text) for sublist in sentences for text in sublist]
    token = [text for sublist in sentences for text in sublist]
    # sentences = [sent.strip() for sent in sentences if sent]
    return token

def Regex_parsers(text,regex_patterns):
    unique_number = {}
    print('regex---->',text)
    for key , pattern in regex_patterns.items():
        finding = re.findall(pattern,text,(re.IGNORECASE|re.MULTILINE))
        try:
            finding = str(finding[0]).strip()
        except:
            pass
        if finding:
            print("---------************{}".format(finding))
            unique_number[key] = [finding]
        else:
            pass
    return unique_number

def Regex_parsers_generator(text,regex_patterns):
    print('regex---->',text)
    for value in text:
        for key , pattern in regex_patterns.items():
            finding = re.findall(pattern,value,(re.IGNORECASE|re.MULTILINE))
            if finding:
                yield key , finding[0]

def msd_data_extractor(list,regex_heading_msd):
    tmp = []
    final = {}
    key = ''
    for i in range(len(list)):
        text = str(list[i])
        if re.findall(regex_heading_msd, text):
            try:
                if key != '':
                    final[key] = '\n'.join(tmp)
                else:
                    pass
                key = text
                tmp.clear()
            except:
                pass
        else:
            if i == len(list) - 1:
                tmp.append(text)
                final[key] = ' '.join(tmp)
            else:
                tmp.append(text)
    return final


def msd_prediction(text):
    model = None
    if os.path.exists(msd_model_location):
        model = joblib.load(msd_model_location)
    else:
        model = msd_model_training()
        print('new model trained')
    # lang_detected = detect(text)
    lang_detected = classify(text)[0]
    # print('lang----->',lang_detected)
    print(text)
    prediction = model.predict(laser.embed_sentences([text],lang=lang_detected))
    probability = model.predict_proba(laser.embed_sentences([text],lang=lang_detected))
    probability[0].sort()
    max_probability = max(probability[0])
    # if (max_probability-(max_probability/2)) > probability[0][-2]:
    if max_probability > 0.60:
        pred_output = prediction[0]
    else:
        pred_output = 'None'
    print('{}-------------->{}'.format(max(probability[0]),pred_output))
    return ({'probability': max(probability[0]), 'output': pred_output, 'actual_output': prediction[0]})

def msd_model_training():
    df = pd.read_excel(msd_input_excel)
    df = df.sample(frac=1)
    X_train_laser = laser.embed_sentences(df['text'], lang='en')
    # mlp = MLPClassifier(hidden_layer_sizes=(125,), solver='adam', activation='tanh', random_state=0, shuffle=True)
    mlp = MLPClassifier(hidden_layer_sizes=(70,),solver='adam',max_iter=500,activation='tanh',random_state=0,shuffle=True)
    # mlp = MLPClassifier(hidden_layer_sizes=(70,),solver='adam',max_iter=300,activation='relu',learning_rate='constant',learning_rate_init=0.001,random_state=0,shuffle=True)
    mlp.fit(X_train_laser, df['category'])
    joblib.dump(mlp,msd_model_location)
    return mlp

# @api_view()
# @permission_classes([IsAuthenticated])
# @authentication_classes([TokenAuthentication])
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
        final = {}
        cate_tmp = {}
        lang_final = set()
        doc_format = os.path.splitext(file_name)[1].lower()
        if doc_format == '.docx':
            # Reading file from storage
            if MODE == 'local':
                file = document_location + file_name
                extracted = text_extraction(file)
            else:
                file = file_name
                extracted = text_extraction(file,method='SMB')
                # file = get_file_smb(r"{}".format(file_name))
            for key,value in extracted.items():
                if "".join(value).strip() != '':
                    result = msd_prediction(key)['output']                          # classifier
                    if result != 'None':
                        if result in final.keys():
                            final[result].extend([val.replace("\n",' ').strip() for val in value])
                        else:
                            final[result] = [val.replace("\n",' ').strip() for val in value]
                    else:
                        pass

            unique = {}
            if 'unique_identifier' in final:
                # unique = Regex_parsers(str(final['unique_identifier']),regex_patterns)
                for key , identifier in Regex_parsers_generator(final['unique_identifier'],regex_patterns):
                    unique[key] = [str(identifier).strip()]
                final.pop('unique_identifier')
            else:
                pass

            for cate , value in final.items():
                if cate in msd_categories_lang:
                    for t in value:
                        if '$$' in t:
                            list_text = t.split('$$')
                            topic = ''
                            for index, text in enumerate(list_text):
                                text = text.replace('$$',' ')
                                if len(str(text).split()) > 2:
                                    text = ' '.join((topic,text)).strip()
                                    topic = ''
                                    lang = detect(text)
                                    lang_final.add(lang)
                                    if cate in cate_tmp:
                                        cate_tmp[cate].append({lang: text})
                                    else:
                                        cate_tmp[cate] = [{lang: text}]
                                else:
                                    topic = ' '.join((topic,text)).strip()
                                    if index == len(list_text)-1:
                                        lang = detect(topic)
                                        lang_final.add(lang)
                                        if cate in cate_tmp:
                                            cate_tmp[cate].append({lang: topic})
                                        else:
                                            cate_tmp[cate] = [{lang: topic}]
                                        topic = ''
                                    else:
                                        pass
                        else:
                            lang = detect(t)
                            lang_final.add(lang)
                            if cate in cate_tmp:
                                cate_tmp[cate].append({lang: t})
                            else:
                                cate_tmp[cate] = [{lang: t}]
                elif cate in msd_categories_lang_exception:
                    # print('^^^^$$$$', value)
                    for t in value:
                        t = t.replace('$$',' ')
                        lang = detect(t)
                        lang_final.add(lang)
                        if cate in cate_tmp:
                            cate_tmp[cate].append({lang: t})
                        else:
                            cate_tmp[cate] = [{lang: t}]
                else:
                    # print('cate------>',cate)
                    cate_tmp[cate] = value
            status = {'status':'1','language': list(lang_final),'file_name':[file_name]}
            extracted_categories = {**status,**cate_tmp,**unique}
            final_json[file_index] = extracted_categories
            # return render(request, 'extractor/doc_result.html', {'result': extracted_categories})
        else:
            status = {'status': '0','file_name': [file_name]}
            final_json[file_index] = status
            # return JsonResponse(status)
    # return Response(final_json)
    return JsonResponse(final_json)


def get_file_smb(file_name):
    data = ''
    try:
        data = smbclient.open_file(r"{}".format(file_name),mode='rb',username=smb_username,password=smb_password)
        print('file found')
    except:
        smbclient.reset_connection_cache()
        data = smbclient.open_file(r"{}".format(file_name), mode='rb',username=smb_username,password=smb_password)
    finally:
        return data

def text_extraction(file,method=None):
    tmp = []
    final = {}
    key = ''
    if method == 'SMB':
        try:
            with smbclient.open_file(r"{}".format(file), mode='rb', username=smb_username, password=smb_password) as f:
                html = mammoth.convert_to_html(f).value
                print('file found')
        except:
            smbclient.reset_connection_cache()
            with smbclient.open_file(r"{}".format(file), mode='rb', username=smb_username, password=smb_password) as f:
                html = mammoth.convert_to_html(f).value
                print('file found')
    else:
        html = mammoth.convert_to_html(file).value
    '''
    soup = BeautifulSoup(html,'html.parser')
    paragraphs = soup.find_all('p')
    # list = [ele.text for ele in paragraphs]
    list = [ele.next for ele in paragraphs]
    '''
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all(['p','li'])
    # -----
    for i, text in enumerate(paragraphs):
        text = str(text)
        if '' in tmp:
            tmp.remove('')
        if re.findall(regex_heading_msd, text):
            try:
                if key and (key not in final):
                    if tmp:
                        final[key] = ['$$'.join(tmp)]
                elif key in final:
                    if tmp:
                        final[key].append('$$'.join(tmp))
                else:
                    pass
                key = re.sub(r'<.*?>', '', text)
                # print(key)
                tmp.clear()
            except:
                pass
        else:
            if i == len(paragraphs) - 1:
                text = text.strip()
                tmp = [t for t in tmp if t]
                if text and not re.findall(r"Panel\s\d", text):
                    text = text.replace('<strong>', '<b>').replace('</strong>', '</b>')
                    text = re.sub(r"<(\/?[^/bems]).*?>", '', text)
                    tmp.append(text)
                if key not in final:
                    if tmp:
                        final[key] = ['$$'.join(tmp)]
                elif key in final:
                    if tmp:
                        final[key].append('$$'.join(tmp))
                else:
                    pass
            else:
                text = text.strip()
                tmp = [t for t in tmp if t]
                if text and not re.findall(r"Panel\s\d", text):  # filter out heading like 'big panel 1'
                    text = text.replace('<strong>', '<b>').replace('</strong>', '</b>')
                    text = re.sub(r"<(\/?[^/bems]).*?>", '', text)
                    tmp.append(text)
    # return final , max(lang,key=lang.count)
    # print(final)
    return final

def extractor(request):
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
        if doc_format == '.xlsx':
            output = Excel_extraction(file_name).main()
            final_json[file_index] = output
        elif doc_format == '.docx':
            output = msd_extraction().main(file_name)
            final_json[file_index] = output
        else:
            print('format not supported')
    return JsonResponse(final_json)


# def dataset_to_mangodb(request):
#     from pymongo import MongoClient
#     client = MongoClient('172.28.42.150',27017)
#     db = client['dataset']
#     collection = db['msd']
#     data = [msd_dataset(category=i['category'], text=i['text'], language_code=i['language_code'],
#                         language=i['language'],
#                         type=i['type']) for i in collection.find({})]
#     if data:
#         msd_dataset.objects.bulk_create(data)
#         return HttpResponse('success')
#     else:
#         return HttpResponse('Failure')

def dataset_to_mangodb(request,django_model,mongo_table):
    from pymongo import MongoClient
    client = MongoClient('172.28.42.150',27017)
    db = client['dataset']
    collection = db[mongo_table]
    data = [django_model(category=i['category'], text=i['text'], language_code=i['language_code'],
                        language=i['language'],
                        category_actual=i['category_actual']) for i in collection.find({})]
    if data:
        msd_dataset.objects.bulk_create(data)
        return HttpResponse('success')
    else:
        return HttpResponse('Failure')

dataset_to_mangodb(msd_contents,'msd_contents')

