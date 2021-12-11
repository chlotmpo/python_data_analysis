from django.shortcuts import render

from django.http import HttpResponse

import pandas as pd

df = pd.read_csv(r'C:\Users\mt181547\OneDrive - De Vinci\Bureau\diabetic_data.csv', sep = ',')

def index(request):
    return HttpResponse(df[0:1000].to_html())