# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Jane Doe", "matriculation": "123456"},
        {"name": "John Smith", "matriculation": "654321"},
        {"name": "Alex Johnson", "matriculation": "789012"},
    ]
    
    projects = [
        {"name": "Home", "url_name": "home:index"},
        {"name": "Home 2", "url_name": "home:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))