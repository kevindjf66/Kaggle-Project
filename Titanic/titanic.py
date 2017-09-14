'''
Project set up on Aug 30, 2017
Created by Kevin S. Deng
'''
#get ready to visualize data, import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#read the csv file
train_data = pd.read_csv('C:\\Users\\Kevin Deng\\Desktop\\Kaggle\\train.csv')
#drop the useless data
train_data = train_data.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis = 1)
#have a look of the datafram to see if there is any NA or ...
train_data.info()
#replace the na in train_data
train_data.Age = train_data.Age.fillna(train_data.Age.mean())
train_data['Fare'] = train_data['Fare'].astype(int)
train_data['Age'] = train_data['Age'].astype(int)
#have a look at what we got
train_data.info()
'''
First, let's do some visualization
'''
#Visualize the connection between PCLASS and SURVIVED
pclass_survived = train_data["Pclass"][train_data["Survived"] == 1]
pclass_unsurvived = train_data["Pclass"][train_data["Survived"] == 0]
plt.figure(num = 1, figsize = (3, 5))
plt.title('Pclass to suviveed', size = 10)
plt.hist(pclass_survived, bins = 7)
#pclass_survived.plot(kind = 'hist', bins = 11, range = [1, 3], figsize = (2, 3))
#Age and survived
facet = sns.FacetGrid(train_data, hue = 'Survived', aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, train_data['Age'].max()))
facet.add_legend()

fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
'''
here comes our prediction
'''


#read the test.csv
test_data = pd.read_csv('C:\\Users\\Kevin Deng\\Desktop\\Kaggle\\test.csv')
test_data.info()
#get rid of the useless data and fill the na
test_data = test_data.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis = 1)
test_data.Age = test_data.Age.fillna(test_data.Age.mean())
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())