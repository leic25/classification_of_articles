import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import numpy as np

df = pd.read_csv('diskretizovano.csv')
x = df.iloc[0:, 0:59]
y = df[' shares_diskretizovano10']

#print(x.head())
#print(y.head())

classes = df[' shares_diskretizovano10'].unique()
n_classes = 10

changes = dict(zip(classes, range(n_classes)))
y = y.replace(changes)

clf = RandomForestClassifier(n_estimators=3, max_depth=40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)

clf.fit(x_train, y_train)
print()
print("Preciznost trening skupa: ", clf.score(x_train, y_train))
print("Preciznost test skupa:  ", clf.score(x_test, y_test))
print()

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

conf = metrics.confusion_matrix(y_test, y_test_pred)
print("Matrica konfuzije:")
print(conf)
print()
print("Izvestaj klasifikacije")
print(metrics.classification_report(y_test, y_test_pred))
