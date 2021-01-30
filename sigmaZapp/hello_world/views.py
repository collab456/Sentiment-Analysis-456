from django.shortcuts import render
from django.http import HttpResponse
from colorama import Fore
import joblib

#all requirment for data preprocessing and building the model
import re
import nltk

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


# Create your views here.
def home(request):
    return render(request,'index.html')


def analyse(request):
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    

    classifier= joblib.load('logistic_classifier.pkl')
    tfidf = joblib.load('tfidf.pkl')

    Review = request.POST['usreview']

    Review = re.sub('[^a-zA-Z]', ' ', Review)
    Review = Review.lower()
    Review = Review.split()
    ptsr = PorterStemmer()
    allow_stopwords = stopwords.words('english')
    allow_stopwords.remove('not')
    Review = [ptsr.stem(word) for word in Review if not word in set(allow_stopwords)]
    Review = ' '.join(Review)
    sortedReview = [Review]
    x_test = tfidf.transform(sortedReview).toarray()
    y_pred = classifier.predict(x_test)

    rvw = str(y_pred[0])
    answer = ''
    if(rvw=='1'):
        answer =  'Happy'+'\U0001f600' 
    else:
        answer = 'Sad'+'\U0001F614'


    #print(Review)

    return render(request, 'index.html',{'answer':answer})

    




