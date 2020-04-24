from django.shortcuts import render,redirect
from django.http import HttpResponse
#from .fake_detect import fakepredict
# Create your views here.
def process_ip_msg(ip_msg):
    text = ip_msg

    import pickle
    infile = open('messages','rb')
    msg = pickle.load(infile)
    infile.close()

    corpus = []
    inp = text.replace(r'[^a-zA-Z]',' ')
    inp = inp.replace(r'\s+',' ')
    inp = inp.rstrip()
    inp = inp.lstrip()
    inp = inp.lower()
    inp = inp.split()
    # print(inp)
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    lmt = WordNetLemmatizer()
    inp_feat = [lmt.lemmatize(word) for word in inp if not word in set(stopwords.words('english'))]
    inp_feat = ' '.join(inp_feat)
    corpus.append(inp_feat)
    print(corpus)
    from sklearn.feature_extraction.text import CountVectorizer
    cv2 = CountVectorizer(max_features = 1700)
    X2 = cv2.fit_transform(msg + corpus).toarray()
    m = X2[-1].reshape(1, -1)
    ################
    from sklearn.externals import joblib
    #joblib.dump(model,'modelpickel.pkl')
    model_from_joblib = joblib.load('modelpickel.pkl')
    result = model_from_joblib.predict(m)

    #result = model.predict(m)
    if result == 1:
        answer = "Genuine"
    else:
        answer = "Fake"
    #print("TEXT INPUT:",text)
    #print()
    #print("ANS:",answer)
    return answer

def news(request):
    print('here')
    submitbutton = request.POST.get("submitbtn")
    ans = ""
    if request.method=='POST':

    
        msg = request.POST['msgtext']
        if submitbutton:
            ans = process_ip_msg(msg)
        else:
            ans = ''
    else:
        msg = " "
        ans = " "
    ###

    #ans = fakepredict(msg)
    context = {'name':'testname','ans':ans,'submitbutton':submitbutton}

    return render(request,"index.html",context)
