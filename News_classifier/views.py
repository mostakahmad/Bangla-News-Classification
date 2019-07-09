from django.shortcuts import render
from django.http import HttpResponse
from . import natural_language_processing
import codecs


def index(request):
    context = {

    }
    return render(request,'index.html',context)

def classify(request):
    if request.method == 'POST':
        query = request.POST.get('news',None)
        print(type(query))
        
        #query=query.encode("utf-8")
        result = natural_language_processing.predict_news(query)
        result = result[0]
        print(result)
        if result == 0:
            return HttpResponse("politis News")
        elif result == 1:
            return HttpResponse(" crime news !!!")
        elif result == 2:
            return HttpResponse(" sports news !!!")
        elif result == 3:
            return HttpResponse(" entertainment news !!!")
        elif result == 4:
            return HttpResponse(" business news !!!")
        elif result == 5:
            return HttpResponse(" lifestyle news !!!")
        

        
    else:
        pass
