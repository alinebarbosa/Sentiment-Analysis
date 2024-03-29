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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, \
                            classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, RFE
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
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
large_matrix_path = '/home/aline/Documentos/Ubiqum/Big data/SentimentAnalysis/data/LargeMatrixMerged.csv'

# read the data and store data in DataFrame
iphone_data = pd.read_csv(iphone_data_path) 
galaxy_data = pd.read_csv(galaxy_data_path)
large_matrix = pd.read_csv(large_matrix_path)

# Create test data
iphone_test = large_matrix.drop(["id"], axis=1)
galaxy_test = large_matrix.drop(["id"], axis=1)

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
iphone_test = iphone_test.drop(["nokiacamunc", 
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
galaxy_test = galaxy_test.drop(["nokiacamunc", 
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
# Engineering the Dependant variable
mapper = {0: 0, 1: 0, 2: 2, 3: 2, 4: 5, 5: 5}
iphone_cor_3v = iphone_corr
iphone_cor_3v['iphonesentiment'] = iphone_cor_3v['iphonesentiment'].map(mapper)
iphone_cor_3v['iphonesentiment'] = pd.Series(iphone_cor_3v['iphonesentiment'],
                                           dtype="category")
iphone_cor_3v.dtypes
iphone_cor_3v['iphonesentiment'].unique()
hist_iphone_3v = px.histogram(iphone_cor_3v, 
                    x="iphonesentiment")
plot(hist_iphone_3v)

galaxy_cor_3v = galaxy_corr
galaxy_cor_3v['galaxysentiment'] = galaxy_cor_3v['galaxysentiment'].map(mapper)
galaxy_cor_3v['galaxysentiment'] = pd.Series(galaxy_cor_3v['galaxysentiment'],
                                           dtype="category")
galaxy_cor_3v.dtypes
galaxy_cor_3v['galaxysentiment'].unique()
hist_galaxy_3v = px.histogram(galaxy_cor_3v, 
                    x="galaxysentiment")
plot(hist_galaxy_3v)

### Over sampling
# Random over sampler
ros = RandomOverSampler(random_state=0)
ros.fit(iphone_corr.iloc[:,0:46], iphone_corr['iphonesentiment'])
iphone_resampled, isent_resampled = ros.sample(iphone_corr.iloc[:,0:46], 
                                               iphone_corr['iphonesentiment'])
iphone_resampled_complete = pd.DataFrame(iphone_resampled)
iphone_resampled_complete['iphonesentiment'] = isent_resampled
hist_iphone_resampled = px.histogram(iphone_resampled_complete,
                                     x='iphonesentiment')
plot(hist_iphone_resampled)

ros.fit(galaxy_corr.iloc[:,0:45], galaxy_corr['galaxysentiment'])
galaxy_resampled, gsent_resampled = ros.sample(galaxy_corr.iloc[:,0:45], 
                                               galaxy_corr['galaxysentiment'])
galaxy_resampled_complete = pd.DataFrame(galaxy_resampled)
galaxy_resampled_complete['galaxysentiment'] = gsent_resampled
hist_galaxy_resampled = px.histogram(galaxy_resampled_complete,
                                     x='galaxysentiment')
plot(hist_galaxy_resampled)

# SMOTE
smote = SMOTE(ratio={0: 100000, 2: 100000, 5: 100000}, random_state=0)
iphone_smote, isent_smote = smote.fit_sample(iphone_cor_3v.iloc[:,0:46], 
                                      iphone_cor_3v['iphonesentiment']) 
iphone_smote_complete = pd.DataFrame(iphone_smote, 
                            columns=list(iphone_cor_3v.iloc[:,0:46].columns))
iphone_smote_complete['iphonesentiment'] = isent_smote
iphone_smote_complete['iphonesentiment'].unique()
hist_iphone_smote = px.histogram(iphone_smote_complete,
                                     x='iphonesentiment')
plot(hist_iphone_smote)

galaxy_smote, gsent_smote = smote.fit_sample(galaxy_cor_3v.iloc[:,0:45], 
                                      galaxy_cor_3v['galaxysentiment']) 
galaxy_smote_complete = pd.DataFrame(galaxy_smote,
                            columns=list(galaxy_cor_3v.iloc[:,0:45].columns))
galaxy_smote_complete['galaxysentiment'] = gsent_smote
galaxy_smote_complete['galaxysentiment'].unique()
hist_galaxy_smote = px.histogram(galaxy_smote_complete,
                                     x='galaxysentiment')
plot(hist_galaxy_smote)

### Under sampling
# Random under sampler
rus = RandomUnderSampler(random_state=0)           #, ratio={0: 30, 1: 20, 2: 60}
rus.fit(iphone_corr.iloc[:,0:46], iphone_corr['iphonesentiment'])
iphone_resampled_under, isent_resampled_under = rus.sample(iphone_corr.iloc[:,0:46], 
                                               iphone_corr['iphonesentiment'])
iphone_resampled_complete_under = pd.DataFrame(iphone_resampled_under)
iphone_resampled_complete_under['iphonesentiment'] = isent_resampled_under
hist_iphone_resampled_under = px.histogram(iphone_resampled_complete_under,
                                     x='iphonesentiment')
plot(hist_iphone_resampled_under)

rus.fit(galaxy_corr.iloc[:,0:45], galaxy_corr['galaxysentiment'])
galaxy_resampled_under, gsent_resampled_under = rus.sample(galaxy_corr.iloc[:,0:45], 
                                               galaxy_corr['galaxysentiment'])
galaxy_resampled_complete_under = pd.DataFrame(galaxy_resampled_under)
galaxy_resampled_complete_under['galaxysentiment'] = gsent_resampled_under
hist_galaxy_resampled_under = px.histogram(galaxy_resampled_complete_under,
                                     x='galaxysentiment')
plot(hist_galaxy_resampled_under)

### Combine over and under sampling
# SMOTETomek
smt = SMOTETomek(ratio='auto')
iphone_smt, isent_smt = smt.fit_sample(iphone_smote, 
                                      isent_smote) 
iphone_smt_complete = pd.DataFrame(iphone_smt)
iphone_smt_complete['iphonesentiment'] = isent_smt
hist_iphone_smt = px.histogram(iphone_smt_complete,
                                     x='iphonesentiment')
plot(hist_iphone_smt)

# ADASYN
adasyn = ADASYN(ratio={0: 30000, 2: 30000, 5: 30000}, random_state=0)
iphone_adasyn, isent_adasyn = adasyn.fit_sample(iphone_smt, 
                                      isent_smt) 
isent_adasyn.dtype
iphone_adasyn_complete = pd.DataFrame(iphone_adasyn)
iphone_adasyn_complete['iphonesentiment'] = pd.Series(isent_adasyn, 
                                           dtype="category")
iphone_adasyn_complete['iphonesentiment'].unique()
hist_iphone_adasyn = px.histogram(iphone_adasyn_complete,
                                     x='iphonesentiment')
plot(hist_iphone_adasyn)

### Learning curves
train_sizes_iphone, train_scores_iphone, validation_scores_iphone = learning_curve(
estimator = rf_classifier,
X = iphone_smote,
y =  isent_smote,
train_sizes = [300, 500, 1000, 3000, 5000, 10000, 15000, 20000, 
               30000, 40000, 50000, 60000, 70000], 
cv = 5,
scoring = 'accuracy')

train_sizes_galaxy, train_scores_galaxy, validation_scores_galaxy = learning_curve(
estimator = rf_classifier,
X = galaxy_corr.iloc[:,0:45],
y = galaxy_corr['galaxysentiment'], 
train_sizes = [300, 500, 1000, 3000, 5000, 10000], 
cv = 5,
scoring = 'accuracy')

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
rf_classifier.feature_importances_

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

##### Data after over sampling
train_iphone_ros, val_iphone_ros, train_isent_ros, val_isent_ros = train_test_split(iphone_resampled, 
                                                                    isent_resampled, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_ros, train_isent_ros)
rf_ros_predictions = rf_classifier.predict(val_iphone_ros)
accuracy_score(val_isent_ros, rf_ros_predictions)
confusion_matrix(val_isent_ros, rf_ros_predictions)
classification_report(val_isent_ros, rf_ros_predictions)
cohen_kappa_score(val_isent_ros, rf_ros_predictions)

#SVM 
svc_model.fit(train_iphone_ros, train_isent_ros)
svc_predictions_ros = svc_model.predict(val_iphone_ros)
accuracy_score(val_isent_ros, svc_predictions_ros)
confusion_matrix(val_isent_ros, svc_predictions_ros)
classification_report(val_isent_ros, svc_predictions_ros)
cohen_kappa_score(val_isent_ros, svc_predictions_ros)

# KNN
knn_model.fit(train_iphone_ros, train_isent_ros)
knn_predictions_ros = knn_model.predict(val_iphone_ros)
accuracy_score(val_isent_ros, knn_predictions_ros)
confusion_matrix(val_isent_ros, knn_predictions_ros)
classification_report(val_isent_ros, knn_predictions_ros)
cohen_kappa_score(val_isent_ros, knn_predictions_ros)

##### Data after SMOTE
train_iphone_smote, val_iphone_smote, train_isent_smote, val_isent_smote = train_test_split(iphone_smote, 
                                                                    isent_smote, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_smote, train_isent_smote)
rf_smote_predictions = rf_classifier.predict(val_iphone_smote)
accuracy_score(val_isent_smote, rf_smote_predictions)
confusion_matrix(val_isent_smote, rf_smote_predictions)
classification_report(val_isent_smote, rf_smote_predictions)
cohen_kappa_score(val_isent_smote, rf_smote_predictions)

#SVM 
svc_model.fit(train_iphone_smote, train_isent_smote)
svc_predictions_smote = svc_model.predict(val_iphone_smote)
accuracy_score(val_isent_smote, svc_predictions_smote)
confusion_matrix(val_isent_smote, svc_predictions_smote)
classification_report(val_isent_smote, svc_predictions_smote)
cohen_kappa_score(val_isent_smote, svc_predictions_smote)

# KNN
knn_model.fit(train_iphone_smote, train_isent_smote)
knn_predictions_smote = knn_model.predict(val_iphone_smote)
accuracy_score(val_isent_smote, knn_predictions_smote)
confusion_matrix(val_isent_smote, knn_predictions_smote)
classification_report(val_isent_smote, knn_predictions_smote)
cohen_kappa_score(val_isent_smote, knn_predictions_smote)

##### Data after under sampling
train_iphone_rus, val_iphone_rus, train_isent_rus, val_isent_rus = train_test_split(iphone_resampled_under, 
                                                                    isent_resampled_under, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_rus, train_isent_rus)
rf_rus_predictions = rf_classifier.predict(val_iphone_rus)
accuracy_score(val_isent_rus, rf_rus_predictions)
confusion_matrix(val_isent_rus, rf_rus_predictions)
classification_report(val_isent_rus, rf_rus_predictions)
cohen_kappa_score(val_isent_rus, rf_rus_predictions)

#SVM 
svc_model.fit(train_iphone_rus, train_isent_rus)
svc_predictions_rus = svc_model.predict(val_iphone_rus)
accuracy_score(val_isent_rus, svc_predictions_rus)
confusion_matrix(val_isent_rus, svc_predictions_rus)
classification_report(val_isent_rus, svc_predictions_rus)
cohen_kappa_score(val_isent_rus, svc_predictions_rus)

# KNN
knn_model.fit(train_iphone_rus, train_isent_rus)
knn_predictions_rus = knn_model.predict(val_iphone_rus)
accuracy_score(val_isent_rus, knn_predictions_rus)
confusion_matrix(val_isent_rus, knn_predictions_rus)
classification_report(val_isent_rus, knn_predictions_rus)
cohen_kappa_score(val_isent_rus, knn_predictions_rus)

##### Data after SMOTETomek
train_iphone_smt, val_iphone_smt, train_isent_smt, val_isent_smt = train_test_split(iphone_smt, 
                                                                    isent_smt, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_smt, train_isent_smt)
rf_smt_predictions = rf_classifier.predict(val_iphone_smt)
accuracy_score(val_isent_smt, rf_smt_predictions)
confusion_matrix(val_isent_smt, rf_smt_predictions)
classification_report(val_isent_smt, rf_smt_predictions)
cohen_kappa_score(val_isent_smt, rf_smt_predictions)

#SVM 
svc_model.fit(train_iphone_smt, train_isent_smt)
svc_predictions_smt = svc_model.predict(val_iphone_smt)
accuracy_score(val_isent_smt, svc_predictions_smt)
confusion_matrix(val_isent_smt, svc_predictions_smt)
classification_report(val_isent_smt, svc_predictions_smt)
cohen_kappa_score(val_isent_smt, svc_predictions_smt)

# KNN
knn_model.fit(train_iphone_smt, train_isent_smt)
knn_predictions_smt = knn_model.predict(val_iphone_smt)
accuracy_score(val_isent_smt, knn_predictions_smt)
confusion_matrix(val_isent_smt, knn_predictions_smt)
classification_report(val_isent_smt, knn_predictions_smt)
cohen_kappa_score(val_isent_smt, knn_predictions_smt)

##### Data after ADASYN
train_iphone_adasyn, val_iphone_adasyn, train_isent_adasyn, val_isent_adasyn = train_test_split(iphone_adasyn, 
                                                                    isent_adasyn, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_iphone_adasyn, train_isent_adasyn)
rf_adasyn_predictions = rf_classifier.predict(val_iphone_adasyn)
accuracy_score(val_isent_adasyn, rf_adasyn_predictions)
confusion_matrix(val_isent_adasyn, rf_adasyn_predictions)
classification_report(val_isent_adasyn, rf_adasyn_predictions)
cohen_kappa_score(val_isent_adasyn, rf_adasyn_predictions)

#SVM 
svc_model.fit(train_iphone_adasyn, train_isent_adasyn)
svc_predictions_adasyn = svc_model.predict(val_iphone_adasyn)
accuracy_score(val_isent_adasyn, svc_predictions_adasyn)
confusion_matrix(val_isent_adasyn, svc_predictions_adasyn)
classification_report(val_isent_adasyn, svc_predictions_adasyn)
cohen_kappa_score(val_isent_adasyn, svc_predictions_adasyn)

# KNN
knn_model.fit(train_iphone_adasyn, train_isent_adasyn)
knn_predictions_adasyn = knn_model.predict(val_iphone_adasyn)
accuracy_score(val_isent_adasyn, knn_predictions_adasyn)
confusion_matrix(val_isent_adasyn, knn_predictions_adasyn)
classification_report(val_isent_adasyn, knn_predictions_adasyn)
cohen_kappa_score(val_isent_adasyn, knn_predictions_adasyn)

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
rf_classifier.feature_importances_

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

##### Data after over sampling
train_galaxy_ros, val_galaxy_ros, train_gsent_ros, val_gsent_ros = train_test_split(galaxy_resampled, 
                                                                    gsent_resampled, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy_ros, train_gsent_ros)
rf_ros_galaxy = rf_classifier.predict(val_galaxy_ros)
accuracy_score(val_gsent_ros, rf_ros_galaxy)
confusion_matrix(val_gsent_ros, rf_ros_galaxy)
classification_report(val_gsent_ros, rf_ros_galaxy)
cohen_kappa_score(val_gsent_ros, rf_ros_galaxy)

#SVM 
svc_model.fit(train_galaxy_ros, train_gsent_ros)
svc_ros_galaxy = svc_model.predict(val_galaxy_ros)
accuracy_score(val_gsent_ros, svc_ros_galaxy)
confusion_matrix(val_gsent_ros, svc_ros_galaxy)
classification_report(val_gsent_ros, svc_ros_galaxy)
cohen_kappa_score(val_gsent_ros, svc_ros_galaxy)

# KNN
knn_model.fit(train_galaxy_ros, train_gsent_ros)
knn_ros_galaxy = knn_model.predict(val_galaxy_ros)
accuracy_score(val_gsent_ros, knn_ros_galaxy)
confusion_matrix(val_gsent_ros, knn_ros_galaxy)
classification_report(val_gsent_ros, knn_ros_galaxy)
cohen_kappa_score(val_gsent_ros, knn_ros_galaxy)

##### Data after under sampling
train_galaxy_rus, val_galaxy_rus, train_gsent_rus, val_gsent_rus = train_test_split(galaxy_resampled_under, 
                                                                    gsent_resampled_under, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy_rus, train_gsent_rus)
rf_rus_galaxy = rf_classifier.predict(val_galaxy_rus)
accuracy_score(val_gsent_rus, rf_rus_galaxy)
confusion_matrix(val_gsent_rus, rf_rus_galaxy)
classification_report(val_gsent_rus, rf_rus_galaxy)
cohen_kappa_score(val_gsent_rus, rf_rus_galaxy)

#SVM 
svc_model.fit(train_galaxy_rus, train_gsent_rus)
svc_rus_galaxy = svc_model.predict(val_galaxy_rus)
accuracy_score(val_gsent_rus, svc_rus_galaxy)
confusion_matrix(val_gsent_rus, svc_rus_galaxy)
classification_report(val_gsent_rus, svc_rus_galaxy)
cohen_kappa_score(val_gsent_rus, svc_rus_galaxy)

# KNN
knn_model.fit(train_galaxy_rus, train_gsent_rus)
knn_rus_galaxy = knn_model.predict(val_galaxy_rus)
accuracy_score(val_gsent_rus, knn_rus_galaxy)
confusion_matrix(val_gsent_rus, knn_rus_galaxy)
classification_report(val_gsent_rus, knn_rus_galaxy)
cohen_kappa_score(val_gsent_rus, knn_rus_galaxy)

##### Data after SMOTE
train_galaxy_smote, val_galaxy_smote, train_gsent_smote, val_gsent_smote = train_test_split(galaxy_smote, 
                                                                    gsent_smote, 
                                                                    random_state = 2)

#Random Forest
rf_classifier.fit(train_galaxy_smote, train_gsent_smote)
rf_smote_galaxy = rf_classifier.predict(val_galaxy_smote)
accuracy_score(val_gsent_smote, rf_smote_galaxy)
confusion_matrix(val_gsent_smote, rf_smote_galaxy)
classification_report(val_gsent_smote, rf_smote_galaxy)
cohen_kappa_score(val_gsent_smote, rf_smote_galaxy)

#SVM 
svc_model.fit(train_galaxy_smote, train_gsent_smote)
svc_smote_galaxy = svc_model.predict(val_galaxy_smote)
accuracy_score(val_gsent_smote, svc_smote_galaxy)
confusion_matrix(val_gsent_smote, svc_smote_galaxy)
classification_report(val_gsent_smote, svc_smote_galaxy)
cohen_kappa_score(val_gsent_smote, svc_smote_galaxy)

# KNN
knn_model.fit(train_galaxy_smote, train_gsent_smote)
knn_smote_galaxy = knn_model.predict(val_galaxy_smote)
accuracy_score(val_gsent_smote, knn_smote_galaxy)
confusion_matrix(val_gsent_smote, knn_smote_galaxy)
classification_report(val_gsent_smote, knn_smote_galaxy)
cohen_kappa_score(val_gsent_smote, knn_smote_galaxy)

"""Apply model - Test"""
# RF - iPhone
rf_classifier.fit(iphone_smote_complete.iloc[:,0:46], 
                  iphone_smote_complete['iphonesentiment'])
rf_iphone_final = rf_classifier.predict(iphone_test)
iphone_test['iphonesentiment'] = pd.Series(rf_iphone_final, dtype="category")
hist_iphone_final = px.histogram(iphone_test, 
                                 x="iphonesentiment")
plot(hist_iphone_final)

# RF - Galaxy
rf_classifier.fit(galaxy_smote_complete.iloc[:,0:45], 
                  galaxy_smote_complete['galaxysentiment'])
rf_galaxy_final = rf_classifier.predict(galaxy_test)
galaxy_test['galaxysentiment'] = pd.Series(rf_galaxy_final, dtype="category")
hist_galaxy_final = px.histogram(galaxy_test, 
                                 x="galaxysentiment")
plot(hist_galaxy_final)
