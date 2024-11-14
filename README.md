# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYAADARSHINI K
RegisterNumber:  212223240126
*/
```
import pandas as pd
data=pd.read_csv("//content/spam.csv", encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![Screenshot 2024-11-14 200805](https://github.com/user-attachments/assets/8dad3e99-bcf8-4e22-9dda-b6baa4ca5614)

![Screenshot 2024-11-14 200814](https://github.com/user-attachments/assets/47d411ce-b426-437f-8dfa-df30a20d0e57)

![Screenshot 2024-11-14 200952](https://github.com/user-attachments/assets/110442a1-4f46-44a3-bb65-d15106fdce75)

![Screenshot 2024-11-14 201011](https://github.com/user-attachments/assets/84dcf9e6-9c3d-42c4-bf11-7709ca08353f)

![Screenshot 2024-11-14 201026](https://github.com/user-attachments/assets/8ba5325d-a114-4a4f-9b2a-f8575056b45f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
