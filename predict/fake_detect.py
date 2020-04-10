import numpy as np
import nltk
import pandas as pd
import sklearn
from .nlptrain import train


def fakepredict(message):
    x_train,x_test,y_train,y_test,msg= train("covid_data.csv")

    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    # Defined models to train
    names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
             "Naive Bayes", "SVM Linear"]

    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier(max_iter = 100),
        MultinomialNB(),
        SVC(kernel = 'linear')
    ]

    models = zip(names, classifiers)
    for name, model in models:
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        cm = confusion_matrix(y_test, pred)
        print(name+" Accuracy:",metrics.accuracy_score(y_test,pred)*100)
        print("Confusion matrix:\n",cm)

    model = MultinomialNB()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    cm = confusion_matrix(y_test, pred)
    print(" Accuracy:",metrics.accuracy_score(y_test,pred)*100)
    print(cm)
    text = message
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
    result = model.predict(m)
    if result == 1:
        answer = "Genuine"
    else:
        answer = "Fake"
    print("TEXT INPUT:",text)
    print()
    print("ANS:",answer)
    if len(message)>1:
        return answer
    else:
        return ""
