import sys
sys.path.append("../../")
from datetime import date, datetime
import time
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.shortcuts import redirect
from django.template import RequestContext
from django.core.files.storage import default_storage
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from threading import Thread
from .models import Document
from .forms import DocumentForm
from HandwrittenTextRecognition_MXNet import demo

filepath = ""


def doStuff(file):
    text = demo.finalfunc(file, "HandwrittenTextRecognition_MXNet")
    time.sleep(2)
    f_name = file.split(".")[0]
    f_name+=".txt"
    f_name = f_name.split("/")[-1]
    f_name="/home/jrmo/Downloads/"+f_name
    with open(f_name, "w+") as out:
        for line in text:
            out.write(line)
    with open(f_name, "r") as f:
        response = HttpResponse(f.read())
        return response


def loading(request):
    t = Thread(target=doStuff, args=(filepath,))
    t.start()
    return list(request, url='loading.html')
    time.sleep(1000)
    #TODO: Add some ML code
    #TODO: Make a done page + auto download

def notjpg(request):
    return list(request, url='notjpg.html')


def list(request, url='list.html'):
    # Handle file upload
    form = DocumentForm(request.POST, request.FILES)
    if form.is_valid() and request.method == 'POST':
        if str(request.FILES['docfile']).split(".")[-1].lower() in ['jpg', 'jpeg']:
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            global filepath
            d = datetime.now()
            filepath = d.strftime("media/documents/%Y/%m/%d/") + str(request.FILES['docfile'])
            return HttpResponseRedirect(reverse('loading'))
        else:
            return HttpResponseRedirect(reverse('notjpg'))
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(request, url, {'documents': documents, 'form': form})
