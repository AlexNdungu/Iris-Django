from turtle import pd
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd

# Create your views here.

def predict(request):
    return render(request, 'predict.html')

def predict_chances(request):

    if request.method == 'POST':

        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        #lets unpickel the model
        model = pd.read_pickle(r"C:\Users\Alex Meta Ndung'u\Documents\Machine Learning\Iris Django\new_model.pickle")

        #Make Predictions
        result = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

        classification = result[0]

        return JsonResponse({'result':classification, 'sepal_length':sepal_length, 'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width},safe=False)