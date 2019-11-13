#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:37:01 2019

#### Sentiment Analysis
# Predict sentiment from different types of devices

@author: Aline Barbosa Alves
"""

# Import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, \
                            classification_report, confusion_matrix, \
                            cohen_kappa_score
from sklearn.model_selection import cross_val_score
import plotly.express as px
import numpy as np
from plotly.offline import plot
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from math import sqrt, pi
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

"""Import Data"""
# Save filepath to variable for easier access
iphone_data_path = '/home/aline/Documentos/Ubiqum/Big data/iphone_smallmatrix_labeled_8d.csv'
galaxy_data_path = '/home/aline/Documentos/Ubiqum/Big data/galaxy_smallmatrix_labeled_9d.csv'

# read the data and store data in DataFrame
iphone_data = pd.read_csv(iphone_data_path) 
galaxy_data = pd.read_csv(galaxy_data_path)

"""Get to know the data"""
# Print a summary of the data
iphone_data.describe()
galaxy_data.describe()

# Columns
iphone_data.columns
galaxy_data.columns

# Missing data
iphone_data.isnull().any().sum()
galaxy_data.isnull().any().sum()

# Basic plot
hist_iphone = px.histogram(iphone_data, 
                    x="iphonesentiment")
plot(hist_iphone)

hist_galaxy = px.histogram(galaxy_data, 
                    x="galaxysentiment")
plot(hist_galaxy)

"""Pre processing"""
# Correlation matrix
iphone_data.corr()
galaxy_data.corr()

c = 0
for i in galaxy_data.corr().abs().unstack().sort_values():
    if (i > 0.9):
        print(i)
        c += 1
        
print(galaxy_data.corr().abs().unstack().sort_values()[3380:3422])

iphone_corr = iphone_data.drop(["nokiacamunc", 
                                "iosperunc", 
                                "nokiaperunc", 
                                "nokiacampos",
                                "samsungdisunc",
                                "ios",
                                "iosperneg",
                                "samsungdisneg",
                                "nokiaperneg",
                                "googleperneg",
                                "nokiadispos",
                                "htcdispos"], axis=1)
    
galaxy_corr = galaxy_data.drop(["nokiacamunc", 
                                "iosperunc", 
                                "nokiaperunc", 
                                "nokiacampos",
                                "samsungdisunc",
                                "ios",
                                "iosperneg",
                                "samsungdisneg",
                                "nokiaperneg",
                                "googleperneg",
                                "nokiadispos",
                                "htcdispos",
                                "sonydisneg"], axis=1)


s = c.unstack()
so = s.sort_values(kind="quicksort")

fig = go.Figure(data=go.Heatmap(z=iphone_data.corr()))
fig.show()

