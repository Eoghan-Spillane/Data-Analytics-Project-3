#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:56:57 2020

@Student name: Eoghan Spillane

@Student ID: R00175214

@Student Course Name: SDH3-C


"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib
import matplotlib.pyplot as plt


def importFile():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # This is used to make print(df) print a larger sample of the dataset to verify my cleaning has worked.

    df = pd.read_csv("people.csv", encoding="ISO-8859-1")
    return df


def cleanData1():
    df = importFile()

    # Removing WhiteSpace and endings
    df['workType'] = df['workType'].str.strip()
    df['education'] = df['education'].str.strip()
    df['native-country'] = df['native-country'].str.strip()
    df['job'] = df['job'].str.strip()
    df['Income'] = df['Income'].str.strip(' \n\t')
    df['marital-status'] = df['marital-status'].str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['relationship'] = df['relationship'].str.strip()
    df.dropna()

    df['Income'] = df['Income'].fillna(0)  # Turning Nan into the most Common Value
    df.loc[df['Income'].str.contains('<=50K', na=False), 'Income'] = 0
    df.loc[df['Income'].str.contains('>50K', na=False), 'Income'] = 1

    # print(df['education'].value_counts()) Most Common = HS-grad
    df['education'] = df['education'].fillna("HS-grad")  # Turning Nan into the most Common Value
    df['education'] = df['education'].str.replace('?', 'Other')
    x = 0
    for y in df['education'].unique():
        df.loc[df['education'] == y, 'education'] = x
        x += 1

    # print(df['workType'].value_counts())  # = Private
    df['workType'] = df['workType'].fillna("Private")  # Turning Nan into the most Common Value
    df['workType'] = df['workType'].str.replace('?', 'Other')
    x = 0
    for y in df['workType'].unique():
        df.loc[df['workType'] == y, 'workType'] = x
        x += 1

    # print(df['job'].value_counts())  # = Prof-specialty
    df['job'] = df['job'].fillna("Prof-specialty")  # Turning Nan into the most Common Value
    df['job'] = df['job'].str.replace('?', 'Salad')
    x = 0
    for y in df['job'].unique():
        df.loc[df['job'] == y, 'job'] = x
        x += 1

    # print(df['job'].isna().any())
    # print(df['workType'].isna().any())
    # print(df['Income'].isna().any())
    # print(df['education'].isna().any())
    # print(df['education'].value_counts())
    # print(df['workType'].value_counts())
    # print(df['job'].value_counts())

    return df


def task1():
    dfAll = cleanData1()
    df = dfAll[dfAll['age'] < 50]
    # Machine Learning Stuff
    X = df[['education', 'workType', 'job']]  # Things to base the predictions around
    y = df[['Income']]  # Things to predict
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print('Training: ', clf.score(X_train, y_train))
    print('Test: ', clf.score(X_test, y_test))

    """
    Training at 33%: 0.8097258229240317
    Testing at 33%: 0.8035405192761605

    Training at 95%: 0.8343815513626834
    Testing at 95%: 0.7873138444567016
    
    When running the code, I noticed that each result has a variance of +-0.03.
    I saw this when running the program multiple times. regardless of testing at 0.95 or 0.333.
    While the training result for 95% was higher the testing result was lower showing that using 33%,
    provide a more consistent pattern to the results that could be favorable
    
    However both 33% and 95% result in a model with a mean accuracy of around 80%
    
    """


def cleanData2():
    df = importFile()

    # Removing WhiteSpace and endings
    df['workType'] = df['workType'].str.strip()
    df['education'] = df['education'].str.strip()
    df['native-country'] = df['native-country'].str.strip()
    df['job'] = df['job'].str.strip()
    df['Income'] = df['Income'].str.strip(' \n\t')
    df['marital-status'] = df['marital-status'].str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['relationship'] = df['relationship'].str.strip()
    df.dropna()

    df['Income'] = df['Income'].fillna(0)  # Turning Nan into the most Common Value
    df.loc[df['Income'].str.contains('<=50K', na=False), 'Income'] = 0
    df.loc[df['Income'].str.contains('>50K', na=False), 'Income'] = 1

    # print(df['workType'].value_counts())  # = Private
    df['workType'] = df['workType'].fillna("Private")  # Turning Nan into the most Common Value

    return df


def task2():
    df = cleanData2()
    twoCols = df[['workType', 'Income']].dropna()

    groups = twoCols.groupby(['workType', 'Income'])

    attribute = "State-gov"

    print('Entropy for', attribute)

    FirstVal = groups.size()[attribute].sum()
    pi = (groups.size()[attribute] / FirstVal)
    log2s = np.log2((groups.size()[attribute] / FirstVal))
    entropy = log2s.multiply(-1 * pi)

    print('Entropy', entropy.sum())

    """
    Entropy for Private = 0.7562417707440523
    Entropy for State-gov = 0.8379148918407011
    
    Both of these values are quite close to 1 which would mean they're "totally random".
    This means that it's quite uncertain that the specific workType has has a huge effect on the on the income etc.
    
    Without also factoring in the something like the job itself in which you could work in the same type of job,
    but be paid significantly more than someone else in the same workType.

    """


def cleanData3():
    df = importFile()

    # Removing WhiteSpace and endings
    df['workType'] = df['workType'].str.strip()
    df['education'] = df['education'].str.strip()
    df['native-country'] = df['native-country'].str.strip()
    df['job'] = df['job'].str.strip()
    df['Income'] = df['Income'].str.strip('. \n\t')
    df['marital-status'] = df['marital-status'].str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['relationship'] = df['relationship'].str.strip()

    x = 0
    for y in df['education'].unique():
        df.loc[df['education'] == y, 'education'] = x
        x += 1

    x = 0
    for y in df['workType'].unique():
        df.loc[df['workType'] == y, 'workType'] = x
        x += 1

    x = 0
    for y in df['job'].unique():
        df.loc[df['job'] == y, 'job'] = x
        x += 1

    average = df['age'].mean()
    df['age'] = df['age'].fillna(int(average))

    return df


def task3():
    s1 = cleanData3()
    s2 = s1[s1['age'] > 30]
    s3 = s2[s2['Gender'] == 'Female']
    df = s3[['education', 'workType', 'age', 'job']]
    df = df.dropna()

    X = df[['education', 'age', 'job']]  # Things to base the predictions around
    y = df[['workType']]  # Things to predict
    y = y.astype('int')
    y = y.values.ravel()

    models = [('DTC', DecisionTreeClassifier()), ('NB', GaussianNB()), ('RFS', RandomForestClassifier())]

    r1 = []
    r2 = []

    results = {}
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy', return_train_score=True)
        results[name] = cv_results

    for models in results:
        print(models)
        print('Training  ', results[models]['train_score'].mean())
        print('Test  ', results[models]['test_score'].mean())

        r1.append(round(results[models]['train_score'].mean(), 4))
        r2.append(round(results[models]['test_score'].mean(), 4))

    """
    Decision Tree classifier
    Training   0.7926206006108463
    Test   0.6954198812296689
    
    Naive bayes (GaussianNB)
    Training   0.7389104797479527
    Test   0.7390637980381892
    
    Random Forest classifier
    Training   0.7925950480364243
    Test   0.7085031134629713
    
    The numbers above are the accuracy scores, they vary quite a bit depending on the classifier thats used.
    The difference between the test and training error is found with the Decision Tree classifier and the Random Forest classifier.
    while the Naive Bayes has only a 0.001 difference between the training and testing while the the other two have a difference of around 0.09.
    """

    labels = ['DCT', 'NB', 'RFS']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, r1, width, label='Training')
    rects2 = ax.bar(x + width / 2, r2, width, label='Testing')

    ax.set_ylabel('Scores')
    ax.set_title('Females overs 30')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def cleanData4():
    df = importFile()

    # Removing WhiteSpace and endings
    df['workType'] = df['workType'].str.strip()
    df['education'] = df['education'].str.strip()
    df['native-country'] = df['native-country'].str.strip()
    df['job'] = df['job'].str.strip()
    df['Income'] = df['Income'].str.strip('. \n\t')
    df['marital-status'] = df['marital-status'].str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['relationship'] = df['relationship'].str.strip()

    x = 0
    for y in df['education'].unique():
        df.loc[df['education'] == y, 'education'] = x
        x += 1

    x = 0
    for y in df['job'].unique():
        df.loc[df['job'] == y, 'job'] = x
        x += 1

    df.loc[df['Income'].str.contains('<=50K', na=False), 'Income'] = 0
    df.loc[df['Income'].str.contains('>50K', na=False), 'Income'] = 1

    average = df['age'].mean()
    df['age'] = df['age'].fillna(int(average))

    return df


def task4():
    s1 = cleanData4()
    s1 = s1.dropna()
    s2 = s1[['Income', 'education', 'job']]
    print("\n Education, Income and Jobs")
    s2 = s2.fillna(s2.mean())
    scalingObj = preprocessing.MinMaxScaler()
    Scaled = scalingObj.fit_transform(s2)
    costs = []  # Storing Results for the Graphs
    for i in range(7):
        kmeans = KMeans(n_clusters=i + 1).fit(Scaled)
        costs.append(kmeans.inertia_)
        print(kmeans.inertia_)  # this line returns cost
    indexs = np.arange(1, 8)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.title("Cluster and K-means inertia displayed via Elbow Method : 1")
    plt.plot(indexs, costs)
    plt.show()

    """
    Using the Elbow method and looking at the data it appears that either 4 is a good number of clusters 
    """

    print("\n Education and Jobs")
    s1 = cleanData4()
    s1 = s1.dropna()
    s2 = s1[['education', 'job']]

    s2 = s2.fillna(s2.mean())
    scalingObj = preprocessing.MinMaxScaler() # normalizing the data so it doesn't get skewed
    Scaled = scalingObj.fit_transform(s2)
    costs = []
    for i in range(7):
        kmeans = KMeans(n_clusters=i + 1).fit(Scaled)
        costs.append(kmeans.inertia_)
        print(kmeans.inertia_)  # this line returns cost
    indexs = np.arange(1, 8)
    plt.xlabel("Number of Clusters")
    plt.title("Cluster and K-means Inertia displayed via Elbow Method : 2")
    plt.ylabel("Distortion")
    plt.plot(indexs, costs)
    plt.show()

    """
    In the above the distortion doesn't reduce as rapidly as previously.
    It starts to level out from the third cluster onwards. 
    This means that having a third attribute most likely helps with increasing accuracy and reducing distortion
    """


task4()
