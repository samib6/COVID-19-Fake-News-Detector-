from django.shortcuts import render,redirect
from django.http import HttpResponse
from .fake_detect import fakepredict
# Create your views here.
def news(request):
    print('here')
    msg = ""
    if request.method=='POST':
        msg = request.POST['msgtext']
    else:
        msg = " "
    ans = fakepredict(msg)
    context = {'name':'testname','ans':ans}

    return render(request,"index.html",context)
