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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, \
                            classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, RFE
import plotly.express as px
from plotly.offline import plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

"""Import Data"""
# Save filepath to variable for easier access
iphone_data_path = '/home/aline/Documentos/Ubiqum/Big data/SentimentAnalysis/data/iphone_smallmatrix_labeled_8d.csv'
galaxy_data_path = '/home/aline/Documentos/Ubiqum/Big data/SentimentAnalysis/data/galaxy_smallmatrix_labeled_9d.csv'

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

# Types
iphone_data.dtypes
iphone_data['iphonesentiment'] = pd.Series(iphone_data['iphonesentiment'],
                                           dtype="category")
galaxy_data.dtypes
galaxy_data['galaxysentiment'] = pd.Series(galaxy_data['galaxysentiment'], 
                                           dtype="category")


# Basic plot
hist_iphone = px.histogram(iphone_data, 
                    x="iphonesentiment")
plot(hist_iphone)

hist_galaxy = px.histogram(galaxy_data, 
                    x="galaxysentiment")
plot(hist_galaxy)

"""Pre processing"""
###### Feature selection
# Correlation matrix
iphone_data.corr()
galaxy_data.corr()

c = 0
for i in galaxy_data.corr().abs().unstack().sort_values():
    if (i > 0.9):
        print(i)
        c += 1
        
print(galaxy_data.corr().abs().unstack().sort_values()[3380:3422])

# Ploting the correlation matrix
fig = go.Figure(data=go.Heatmap(z=iphone_data.corr()))
fig.show()

# Drop columns with high correlation (> 0.9)
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

# Exploring feature variance
selector = VarianceThreshold()
iphone_var = selector.fit_transform(iphone_data)
iphone_var = pd.DataFrame(iphone_var, columns=list(iphone_data.columns))
galaxy_var = selector.fit_transform(galaxy_data)
galaxy_var = pd.DataFrame(galaxy_var, columns=list(galaxy_data.columns))

# Exploring recursive feature elimination
rf_classifier = RandomForestClassifier(n_estimators=400)
rfe_selector = RFE(rf_classifier)
rfe_fit = rfe_selector.fit(iphone_data.iloc[:,0:58], 
                           iphone_data['iphonesentiment'])
rfe_fit.support_
rfe_fit.ranking_
iphone_rfe = iphone_data.drop(iphone_data.columns[rfe_fit.ranking_],axis=1)

rfe_fit_galaxy = rfe_selector.fit(galaxy_data.iloc[:,0:58], 
                                  galaxy_data['galaxysentiment'])
rfe_fit_galaxy.support_
rfe_fit_galaxy.ranking_
galaxy_rfe = galaxy_data.drop(galaxy_data.columns[rfe_fit_galaxy.ranking_],
                              axis=1)

##### Feature engineering
# PCA


"""Models - iPhone"""
##### Out of the box
train_iphone, val_iphone, train_isent, val_isent = train_test_split(iphone_data.iloc[:,0:58], 
                                                                    iphone_data['iphonesentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone, train_isent)
rf_classifier_predictions = rf_classifier.predict(val_iphone)
accuracy_score(val_isent, rf_classifier_predictions)
confusion_matrix(val_isent, rf_classifier_predictions)
classification_report(val_isent, rf_classifier_predictions)
cohen_kappa_score(val_isent, rf_classifier_predictions)

#SVM 
svc_model = SVC()
svc_model.fit(train_iphone, train_isent)
svc_predictions = svc_model.predict(val_iphone)
accuracy_score(val_isent, svc_predictions)
confusion_matrix(val_isent, svc_predictions)
classification_report(val_isent, svc_predictions)
cohen_kappa_score(val_isent, svc_predictions)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_iphone, train_isent)
knn_predictions = knn_model.predict(val_iphone)
accuracy_score(val_isent, knn_predictions)
confusion_matrix(val_isent, knn_predictions)
classification_report(val_isent, knn_predictions)
cohen_kappa_score(val_isent, knn_predictions)

##### Data without high correlated features
train_iphone_cor, val_iphone_cor, train_isent_cor, val_isent_cor = train_test_split(iphone_corr.iloc[:,0:46], 
                                                                    iphone_corr['iphonesentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_cor, train_isent_cor)
rf_corr_predictions = rf_classifier.predict(val_iphone_cor)
accuracy_score(val_isent_cor, rf_corr_predictions)
confusion_matrix(val_isent_cor, rf_corr_predictions)
classification_report(val_isent_cor, rf_corr_predictions)
cohen_kappa_score(val_isent_cor, rf_corr_predictions)

#SVM 
svc_model.fit(train_iphone_cor, train_isent_cor)
svc_predictions_cor = svc_model.predict(val_iphone_cor)
accuracy_score(val_isent_cor, svc_predictions_cor)
confusion_matrix(val_isent_cor, svc_predictions_cor)
classification_report(val_isent_cor, svc_predictions_cor)
cohen_kappa_score(val_isent_cor, svc_predictions_cor)

# KNN
knn_model.fit(train_iphone_cor, train_isent_cor)
knn_predictions_cor = knn_model.predict(val_iphone_cor)
accuracy_score(val_isent_cor, knn_predictions_cor)
confusion_matrix(val_isent_cor, knn_predictions_cor)
classification_report(val_isent_cor, knn_predictions_cor)
cohen_kappa_score(val_isent_cor, knn_predictions_cor)

##### Data after recursive feature elimination
train_iphone_rfe, val_iphone_rfe, train_isent_rfe, val_isent_rfe = train_test_split(iphone_rfe.iloc[:,0:28], 
                                                                    iphone_rfe['iphonesentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_rfe, train_isent_rfe)
rf_rfe_predictions = rf_classifier.predict(val_iphone_rfe)
accuracy_score(val_isent_rfe, rf_rfe_predictions)
confusion_matrix(val_isent_rfe, rf_rfe_predictions)
classification_report(val_isent_rfe, rf_rfe_predictions)
cohen_kappa_score(val_isent_rfe, rf_rfe_predictions)

#SVM 
svc_model.fit(train_iphone_rfe, train_isent_rfe)
svc_predictions_rfe = svc_model.predict(val_iphone_rfe)
accuracy_score(val_isent_rfe, svc_predictions_rfe)
confusion_matrix(val_isent_rfe, svc_predictions_rfe)
classification_report(val_isent_rfe, svc_predictions_rfe)
cohen_kappa_score(val_isent_rfe, svc_predictions_rfe)

# KNN
knn_model.fit(train_iphone_rfe, train_isent_rfe)
knn_predictions_rfe = knn_model.predict(val_iphone_rfe)
accuracy_score(val_isent_rfe, knn_predictions_rfe)
confusion_matrix(val_isent_rfe, knn_predictions_rfe)
classification_report(val_isent_rfe, knn_predictions_rfe)
cohen_kappa_score(val_isent_rfe, knn_predictions_rfe)

"""Models - Galaxy"""
##### Out of the box
train_galaxy, val_galaxy, train_gsent, val_gsent = train_test_split(galaxy_data.iloc[:,0:58], 
                                                                    galaxy_data['galaxysentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy, train_gsent)
rf_predictions_galaxy = rf_classifier.predict(val_galaxy)
accuracy_score(val_gsent, rf_predictions_galaxy)
confusion_matrix(val_gsent, rf_predictions_galaxy)
classification_report(val_gsent, rf_predictions_galaxy)
cohen_kappa_score(val_gsent, rf_predictions_galaxy)

#SVM 
svc_model.fit(train_galaxy, train_gsent)
svc_predictions_galaxy = svc_model.predict(val_galaxy)
accuracy_score(val_gsent, svc_predictions_galaxy)
confusion_matrix(val_gsent, svc_predictions_galaxy)
classification_report(val_gsent, svc_predictions_galaxy)
cohen_kappa_score(val_gsent, svc_predictions_galaxy)

# KNN
knn_model.fit(train_galaxy, train_gsent)
knn_predictions_galaxy = knn_model.predict(val_galaxy)
accuracy_score(val_gsent, knn_predictions_galaxy)
confusion_matrix(val_gsent, knn_predictions_galaxy)
classification_report(val_gsent, knn_predictions_galaxy)
cohen_kappa_score(val_gsent, knn_predictions_galaxy)

##### Data without high correlated features
train_galaxy_cor, val_galaxy_cor, train_gsent_cor, val_gsent_cor = train_test_split(galaxy_corr.iloc[:,0:45], 
                                                                    galaxy_corr['galaxysentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy_cor, train_gsent_cor)
rf_corr_galaxy = rf_classifier.predict(val_galaxy_cor)
accuracy_score(val_gsent_cor, rf_corr_galaxy)
confusion_matrix(val_gsent_cor, rf_corr_galaxy)
classification_report(val_gsent_cor, rf_corr_galaxy)
cohen_kappa_score(val_gsent_cor, rf_corr_galaxy)

#SVM 
svc_model.fit(train_galaxy_cor, train_gsent_cor)
svc_cor_galaxy = svc_model.predict(val_galaxy_cor)
accuracy_score(val_gsent_cor, svc_cor_galaxy)
confusion_matrix(val_gsent_cor, svc_cor_galaxy)
classification_report(val_gsent_cor, svc_cor_galaxy)
cohen_kappa_score(val_gsent_cor, svc_cor_galaxy)

# KNN
knn_model.fit(train_galaxy_cor, train_gsent_cor)
knn_cor_galaxy = knn_model.predict(val_galaxy_cor)
accuracy_score(val_gsent_cor, knn_cor_galaxy)
confusion_matrix(val_gsent_cor, knn_cor_galaxy)
classification_report(val_gsent_cor, knn_cor_galaxy)
cohen_kappa_score(val_gsent_cor, knn_cor_galaxy)

##### Data after recursive feature elimination
train_galaxy_rfe, val_galaxy_rfe, train_gsent_rfe, val_gsent_rfe = train_test_split(galaxy_rfe.iloc[:,0:28], 
                                                                    galaxy_rfe['galaxysentiment'], 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy_rfe, train_gsent_rfe)
rf_rfe_galaxy = rf_classifier.predict(val_galaxy_rfe)
accuracy_score(val_gsent_rfe, rf_rfe_galaxy)
confusion_matrix(val_gsent_rfe, rf_rfe_galaxy)
classification_report(val_gsent_rfe, rf_rfe_galaxy)
cohen_kappa_score(val_gsent_rfe, rf_rfe_galaxy)
rf_classifier.feature_importances_

#SVM 
svc_model.fit(train_galaxy_rfe, train_gsent_rfe)
svc_rfe_galaxy = svc_model.predict(val_galaxy_rfe)
accuracy_score(val_gsent_rfe, svc_rfe_galaxy)
confusion_matrix(val_gsent_rfe, svc_rfe_galaxy)
classification_report(val_gsent_rfe, svc_rfe_galaxy)
cohen_kappa_score(val_gsent_rfe, svc_rfe_galaxy)

# KNN
knn_model.fit(train_galaxy_rfe, train_gsent_rfe)
knn_rfe_galaxy = knn_model.predict(val_galaxy_rfe)
accuracy_score(val_gsent_rfe, knn_rfe_galaxy)
confusion_matrix(val_gsent_rfe, knn_rfe_galaxy)
classification_report(val_gsent_rfe, knn_rfe_galaxy)
cohen_kappa_score(val_gsent_rfe, knn_rfe_galaxy)
