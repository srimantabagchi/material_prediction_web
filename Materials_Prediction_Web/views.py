from django.shortcuts import render
from tensorflow.keras.models import load_model
import pandas as pd
from pickle import load
from sklearn.preprocessing import StandardScaler
import numpy as np

# our home page view
def home(request):    
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(HMA_C,HMA_MN,HMA_S,HMA_P,HMA_SI,HMA_TI,HMA_CR,BLOW_DUR,HMWT,SCP,HMTEMP,LIME,DOLO,ORE,OXY):

    #Import machine learning model
    model = load_model('/mnt/c/projects/data_science/Materials_Prediction_Web/Materials_Prediction_Web/ANN_Model_B.h5')

    print("In here")

    #Import the standardization scaler
    scaler = load(open('/mnt/c/projects/data_science/Materials_Prediction_Web/Materials_Prediction_Web/scaler.pkl', 'rb'))

    print(HMA_C,HMA_MN,HMA_S,HMA_P,HMA_SI,HMA_TI,HMA_CR,BLOW_DUR,HMWT,SCP,HMTEMP,LIME,DOLO,ORE,OXY)
    #Transform the input variables by using the scaler
    x = scaler.transform(np.array([HMA_C,HMA_MN,HMA_S,HMA_P,HMA_SI,HMA_TI,HMA_CR,BLOW_DUR,HMWT,SCP,HMTEMP,LIME,DOLO,ORE,OXY]).reshape(1,-1))

    #Use machine learning model to predict endpoint temperature
    pred = model.predict(x)
    print('The endpoint temperature for this heat is:',pred)

    return pred[0][0]
        

# our result page view
def result(request):
    HMA_C = float(request.GET['HMA_C'])
    HMA_MN = float(request.GET['HMA_MN'])
    HMA_S = float(request.GET['HMA_S'])
    HMA_P = float(request.GET['HMA_P'])
    HMA_SI = float(request.GET['HMA_SI'])
    HMA_TI = float(request.GET['HMA_TI'])
    HMA_CR = float(request.GET['HMA_CR'])
    BLOW_DUR = float(request.GET['BLOW_DUR'])
    HMWT = float(request.GET['HMWT'])
    SCP = float(request.GET['SCP'])
    HMTEMP = float(request.GET['HMTEMP'])
    LIME = float(request.GET['LIME'])
    DOLO = float(request.GET['DOLO'])
    ORE = float(request.GET['ORE'])
    OXY = float(request.GET['OXY'])

    result = getPredictions(HMA_C,HMA_MN,HMA_S,HMA_P,HMA_SI,HMA_TI,HMA_CR,BLOW_DUR,HMWT,SCP,HMTEMP,LIME,DOLO,ORE,OXY)

    return render(request, 'result.html', {'result':result})