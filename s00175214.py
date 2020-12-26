#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:56:57 2020

@Student name: Eoghan Spillane

@Student ID: R00175214

@Student Course Name: SDH3-C


"""

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def importFile():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
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

    # Training at 33%: 0.8097258229240317
    # Testing at 33%: 0.8035405192761605

    # Training at 95%: 0.8343815513626834
    # Testing at 95%: 0.7873138444567016

    """
    Training at 33%: 0.8097258229240317
    Testing at 33%: 0.8035405192761605

    Training at 95%: 0.8343815513626834
    Testing at 95%: 0.7873138444567016
    
    When running the code, I noticed that each result has a variance of +-0.03.
    I saw this when running the program multiple times. regardless of testing at 0.95 or 0.333.
    
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
    twoCols = df[['Pclass', 'Survived']].dropna()

    groups = twoCols.groupby(['Pclass', 'Survived'])

    print(groups.size())

    # uniques = np.unique(twoCols['Pclass'])

    attribute_value = 1

    print('Entropy for', attribute_value)

    sumVal1 = groups.size()[attribute_value].sum()
    print(sumVal1, groups.size()[attribute_value])
    # print(sumVal1, groups.size()[first_value])

    pi = (groups.size()[attribute_value] / sumVal1)
    # print(pi)
    log2s = np.log2((groups.size()[attribute_value] / sumVal1))

    entropies = log2s.multiply(-1 * pi)

    print('Entropy', entropies.sum())


def task3():
    """TODO"""


def task4():
    """TODO"""


task1()
