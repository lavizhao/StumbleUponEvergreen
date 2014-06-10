#coding: utf-8

from read_conf import config
import csv
import numpy as np
from sklearn.preprocessing import Imputer

dp = config("../conf/dp.conf")

f = open(dp["raw_train"])

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

reader = csv.reader(f,delimiter='\t')

a = 0
train = []
for line in reader :
    if a == 0:
        a += 1
        continue
    temp = []
    line = line[5:]
    line = line[:-1]
    for item in line:
        if item == "?":
            temp.append(np.nan)
        else :
            temp.append(float(item))
    train.append(temp)
    a += 1

imp.fit(train)
train = imp.transform(train)
for i in range(10):
    print train[i]
