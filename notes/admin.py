from django.contrib import admin

from . import models

# Register your models here.
class NotesAdmin(admin.ModelAdmin): # inherit from admin.ModelAdmin
    list_display = ('title')

admin.site.register(models.Notes, NotesAdmin)