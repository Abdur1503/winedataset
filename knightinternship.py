# -*- coding: utf-8 -*-
"""
Created on Mon May 11 01:00:58 2020

@author: abdur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
#nltk.download('stopwords')
#for removing preposition ,pronoun etc
from nltk.corpus import stopwords

#for stemming
from nltk.stem.porter import PorterStemmer
pt=PorterStemmer()

#for remove emoji number and other thing@user re stands regulgua expression 
import re

dataset=pd.read_csv('train.csv')
dataset.isnull().sum()

#leaving username and review description as no relation with variety
#leaving designation and region2 as too many outliers
X=dataset.iloc[:,[1,2,5,6,7,8,10]].values #feature matrix 
y=dataset.iloc[:,[-1]].values


vintage=dataset['review_title']

#to store vintage only
vin_temp=[]

for i in range(82657):
    #taking out the year
    temp=re.sub('[^0-9]',' ',vintage[i])
    tweet=temp.split()
    if len(tweet)==0:
        temp='nan'
    else:
        
        temp=int(tweet[0])
    vin_temp.append(temp)

del vintage,temp,tweet,i



#Handling missing values of country,province and region1 and also storing vintage back to feature matrix
temp=pd.DataFrame(X[:,[0,1,4,5]])
temp[1]=vin_temp

temp[0].value_counts() #to find the mode i.e US
temp[2].value_counts() #to find the mode i.e California
temp[3].value_counts() #to find the mode i.e  Napa Valley


#filling nan with mode except in vintage it will be filled by mean
temp[0]=temp[0].fillna('US')     
temp[2]=temp[2].fillna('California')
temp[3]=temp[3].fillna('Napa Valley')

temp.isnull().sum()

X[:,[0,1,4,5]]=temp

del temp,vin_temp


#handling the missing values with mean for vintage and price
from sklearn.preprocessing import Imputer
sim=Imputer(missing_values="NaN", strategy="mean")

X[:,[1,3]]=sim.fit_transform(X[:,[1,3]])



#handling categorical values

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()

X[:,0]=lab.fit_transform(X[:,0])
X[:,4]=lab.fit_transform(X[:,4])
X[:,5]=lab.fit_transform(X[:,5])
X[:,6]=lab.fit_transform(X[:,6])

y=lab.fit_transform(y)

#checking y
lab.classes_


#Sparse matrix for feature matrix all strings as the aftr having number can be thought as of related
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[0,4,5,6])
X=one.fit_transform(X)
X=X.toarray()

#Scaling now the feature matrix
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=25)#incresase precission
dtf.fit(X,y)
dtf.score(X,y)

test=pd.readcsv("test.csv")
X_test=test.iloc[:,[1,2,5,6,7,8,10]].values #feature matrix 

y_test=dtf.predict(X_test)

from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)




















