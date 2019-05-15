from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

bankdata = pd.read_csv(r"train.csv", encoding='UTF-8')
X = bankdata.drop('class', axis=1)#
y = bankdata['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#clf = tree.DecisionTreeClassifier() if you want to use decision tree
clf = RandomForestClassifier(n_estimators=70)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))              #
print(classification_report(y_test,y_pred))