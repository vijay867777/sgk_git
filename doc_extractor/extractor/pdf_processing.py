import textract
import smbclient

from .local_constants import *

input_pdf = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/QRD_English.pdf"
input_pdf = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/11.pdf"

input_doc = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/issue_1.docx"

text = textract.process(input_pdf,method='pdfminer')
text = textract.process(input_pdf,encoding='utf-8')


text = textract.process(input_doc,encoding='utf-8')

text1 = text.decode('utf-8')

text

text1.splitlines()

paragraphs = str(text1).split('\n\n')

for para in paragraphs:
    print('--------'*5)
    print(para)
    print('--------' * 5)


import pdfplumber

plumber = pdfplumber.open(input_pdf)

plum_text = plumber.pages[0].extract_text()

plum_paragraphs = plum_text.split()

plum_text.splitlines()


from depdf import DePDF
from bs4 import BeautifulSoup

depdf_ext = DePDF.load(input_pdf)

html = depdf_ext.to_html

soup = BeautifulSoup(html,parser='html')

for tag in soup.find_all('p'):
    print('----'*5)
    print(tag.text)
    print('----'*5)

from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text_to_fp
import sys

if sys.version_info > (3, 0):
    from io import StringIO
else:
    from io import BytesIO as StringIO

output_string = StringIO()
with open(input_pdf,'rb') as inp:
    extract_text_to_fp(inp,output_string,laparams=LAParams(),output_type='text',codec=None)

print(output_string.getvalue().strip())

class pdf_extraction:
    def __init__(self,file_location):
        self.file_location = file_location

    def extract_pdf(self):
        if self.file_location.startswith('\\'):
            try:
                with smbclient.open_file(r"{}".format(self.file_location), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    pdf_extracted = pdfplumber.open(f)
                    print('file found')
            except:
                smbclient.reset_connection_cache()
                with smbclient.open_file(r"{}".format(self.file_location), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    pdf_extracted = pdfplumber.open(f)
                    print('file found')
            finally:
                smbclient.reset_connection_cache()

        else:
            file = document_location + self.file_location
            print(file)
            pdf_extracted = pdfplumber.open(file)

        return pdf_extracted

