import requests
import pandas as pd

from numpy import mean
from numpy import std

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    print('[INFO] START GENREATE KMERS LIST')

    three_mers = [  # AAA, AAC, AAG ...
        f'{a}{b}{c}{d}{e}{f}'
        #f"{a}{b}{c}{d}{e}{f}{g}"
        for a in ["A", "C", "U", "G"]
        for b in ["A", "C", "U", "G"]
        for c in ["A", "C", "U", "G"]
        for d in ["A", "C", "U", "G"]
        for e in ["A", "C", "U", "G"]
        for f in ["A", "C", "U", "G"]
        #for g in ["A", "C", "U", "G"]
    ]

    print('[INFO] FINISH KMER LIST')
    print('[INFO] START READING FILE')
    # Read the data
    with open("ELAVL1_PARCLIP.txt") as f:
        data_text = f.read().split("\n>")

    data = []

    print('[INFO] START GETING TARGETS')
    for read in data_text:
        dic = {'target': read.split('\n')[0].split('|')[-1]}
        seq = read.split('\n')[-1]
        for kmer in three_mers:
            dic[kmer] = seq.count(kmer)
        data.append(dic)
    # convert to pandas dataframe
    df = pd.DataFrame(data)

    print('[INFO] FINISH COUNTING')

    X = df.iloc[:, 1:]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=100
    )

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    dtc_pred = dtc.predict(X_test)
    print('Metrics for decision tree:')
    print(f'AUROC: {metrics.roc_auc_score(y_test, dtc_pred)}')
    print(f'Accuracy: {metrics.accuracy_score(y_test, dtc_pred)}')
    print(f'Precision: {metrics.precision_score(y_test, dtc_pred)}')
    print(f'Recall: {metrics.recall_score(y_test, dtc_pred)}')
    print(f'F1: {metrics.f1_score(y_test, dtc_pred)}')
    print('\n\n')

    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train)

    knn_pred = knn.predict(X_test)


    print('Metrics for k neighbors:')
    print(f'AUROC: {metrics.roc_auc_score(y_test, knn_pred)}')
    print(f'Accuracy: {metrics.accuracy_score(y_test, knn_pred)}')
    print(f'Precision: {metrics.precision_score(y_test, knn_pred)}')
    print(f'Recall: {metrics.recall_score(y_test, knn_pred)}')
    print(f'F1: {metrics.f1_score(y_test, knn_pred)}')