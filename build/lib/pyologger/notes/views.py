from django.shortcuts import render

# Create your views here.

from .models import Notes

def list(request): # making a function to view all notes
    all_notes = Notes.objects.all()
    return render(request, 'notes/notes_list.html', {'notes': all_notes})
