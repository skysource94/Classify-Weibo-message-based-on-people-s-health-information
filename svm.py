import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

bankdata = pd.read_csv(r"train.csv", encoding='UTF-8')
X = bankdata.drop('class', axis=1)#
y = bankdata['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



