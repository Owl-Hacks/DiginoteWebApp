from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.shortcuts import redirect
from django.template import RequestContext
from django.core.files.storage import default_storage
from django.http import HttpResponseRedirect
from django.urls import reverse

from .models import Document
from .forms import DocumentForm

def doStuff(file):
    path = default_storage.save('tmp/' + str(file), ContentFile(file.read()))
    print(path)

def loading(request):
    return list(request, url='loading.html')

def notjpg(request):
    return list(request, url='notjpg.html')

def list(request, url='list.html'):
    # Handle file upload
    form = DocumentForm(request.POST, request.FILES)
    if form.is_valid() and request.method == 'POST':
        if str(request.FILES['docfile']).split(".")[-1].lower() in ['jpg', 'jpeg']:
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            return HttpResponseRedirect(reverse('loading'))
        else:
            return HttpResponseRedirect(reverse('notjpg'))
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(request, url, {'documents': documents, 'form': form})