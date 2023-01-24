import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv("Iris.csv")
print(df)
df1=df.drop('Id',axis=1)
df1.drop_duplicates()
print(df1.describe())
print(df1.info)
print(df1['Species'].value_counts())
print(df1.isnull().sum())


# Data Visualization
sns.histplot(x="SepalLengthCm", hue="Species", data=df1)
plt.show()
sns.histplot(x="SepalWidthCm", hue="Species", data=df1)
plt.show()
sns.histplot(x="PetalLengthCm", hue="Species", data=df1)
plt.show()
sns.histplot(x="PetalWidthCm", hue="Species", data=df1)
plt.show()


S=df1["SepalLengthCm"]
R=df1["SepalWidthCm"]
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm ")
plt.scatter(S,R,)
plt.show()
P=df1["PetalLengthCm"]
Q=df1["PetalWidthCm"]
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.scatter(P,Q)
plt.show()




print(df.corr())
sns.heatmap(df.corr(),cmap="BuGn",annot=True)
plt.show()
X=df1.iloc[:,:-1]
Y=df1.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=0)
GaussNB=GaussianNB()
GaussNB.fit(X_train,Y_train)
GaussNB_predict=GaussNB.predict(X_test)
Dtree=DecisionTreeClassifier(random_state=0)
Dtree.fit(X_train,Y_train)
Dtree_predict=Dtree.predict(X_test)
print(Dtree_predict)
svm_clf=svm.SVC(kernel="linear")
svm_clf.fit(X_train,Y_train)
svm_clf_predict=svm_clf.predict(X_test)
print(accuracy_score(Y_test,svm_clf_predict)*100)