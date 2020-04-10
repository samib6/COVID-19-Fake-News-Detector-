import numpy as np
import nltk
import pandas as pd
import sklearn

def train(csv_name):
    ds = pd.read_csv(csv_name)
    df = pd.DataFrame(ds,columns=['Head','Message','Type'])
    classes = df['Type']
    print("This",classes.value_counts())

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    Y = encoder.fit_transform(classes)

    messages = df["Message"]
    msg = messages.str.replace(r'[^a-zA-Z]',' ')
    msg = msg.str.replace(r'\s+',' ')
    msg = msg.str.rstrip()
    msg = msg.str.lstrip()
    msg = msg.str.lower()

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    msg = msg.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    from nltk.stem import WordNetLemmatizer
    lmt = WordNetLemmatizer()
    msg = msg.apply(lambda x: ' '.join(lmt.lemmatize(term) for term in x.split()))

    from nltk.tokenize import word_tokenize
    all_words = []

    for any in msg:
        words = word_tokenize(any)
        for w in words:
            all_words.append(w)

    all_words = nltk.FreqDist(all_words)
    l=len(all_words)
    word_features = list(all_words.keys())[:l]

    def find_features(message,word_features):
        words = word_tokenize(message)
        features = {}
        for word in word_features:
            features[word] = (word in words)
        return features

    # features = find_features(msg[3])
    # print(features)
    # for key, value in features.items():
    #     if value == True:
    #         print('Key:',key)

    # zipped = zip(msg,Y)
    # # print(list(zipped),type(zipped))
    # zipped = list(zipped)
    # np.random.seed(1)
    # np.random.shuffle(zipped)

    # print(zipped)
    # featuresets = [(find_features(text,word_features), label) for (text, label) in zipped]
    # for text,label in list(zipped):
    #     print('look:',text,label)
    # print(featuresets)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1700)
    x = cv.fit_transform(msg).toarray()

    from sklearn import model_selection
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,Y, test_size = 0.25, random_state=1)

    return x_train,x_test,y_train,y_test,msg
