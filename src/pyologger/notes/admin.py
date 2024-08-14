from django.contrib import admin

from . import models

# Register your models here.
class NotesAdmin(admin.ModelAdmin): # inherit from admin.ModelAdmin
    list_display = ('title','text','created') # display these fields in the admin panel)

admin.site.register(models.Notes, NotesAdmin)