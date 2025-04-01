from IPython import get_ipython
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

%matplotlib inline
columns=['sepal length','sepal width','petal length','petal width','class_labels']
df = pd.read_csv('iris.data',names=columns)
df.head()
#visualize the whole dataset
sns.pairplot(df, hue='class_labels', vars=['sepal length', 'sepal width', 'petal length', 'petal width'], diag_kind='hist')
# Explicitly specify the numerical columns for plotting using the 'vars' parameter.
# Use 'hist' for the diagonal plots to handle categorical data.
#separate the features and targett
data=df.values
# Exclude the first row (column names) - Indexing starts from 0
x=data[1:,0:4]  # This line was causing the error
y=data[1:,4]

# Convert to float AFTER removing the header row
x = x.astype(float)  # Corrected line to convert to float after removing header row
print(df)
print(data)
#split the data to train and test dataset.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_test)
model_svc=SVC() # this is defined in cell 24
model_svc.fit(x_train,y_train) # make sure this line is executed before the prediction in cell 25
prediction1 = model_svc.predict(x_test)
print(accuracy_score(y_test,prediction1)*100)
for i in range(len(prediction1)):
  print(y_test[i],prediction1[i])
