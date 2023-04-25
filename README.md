## cs5293sp23-project2
### Project2 - Text Analytics - Cuisine Prediction
### Vasu Deva Sai Nadha Reddy Janapala

The following are the libraries used in this project:
```
import json
import pandas as pd
import argparse
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
```

For this project, I utilized *logistic regression* by importing relevant libraries using *scikit-learn*. The imported libraries include **train_test_split**, **LogisticRegression**, **accuracy_score**, **TfidfVectorizer**, and **cosine_similarity**.

### Step1:
In the main method, I have included two parameters - N and ingredient, which are the command line arguments. These inputs are expected from the user using the following command - 
```
pipenv run python project2.py --N 5 --ingredient "dry white wine" --ingredient "olive oil" --ingredient "garlic cloves" --ingredient lemon
```
After meeting the requirement, we move on to **step 2**.

### Step 2:
Now, the arguments are passed to the *cuisin_predict()* method. After passing the arguments, I parsed the information to local variables.
Created pickle paths, to store the model data, preprocessed information in the form of objects. Below is the block of codes I used - 
```
model_path = Path.cwd() / "logistic.pkl"
X_path = Path.cwd() / "X_tra.pkl"
vectorizer_path = Path.cwd() / "vectorizer.pkl"
dataframe_path = Path.cwd() / "dataframe.pkl"
```
Under if block:
```
with open(model_path.absolute(), "wb") as model_f, open(X_path.absolute(), "wb") as x_f, open(vectorizer_path.absolute(), "wb") as vector_f, open(dataframe_path.absolute(), "wb") as df_f:
  pickle.dump(model, model_f)
  pickle.dump(X, x_f)
  pickle.dump(vectorizer, vector_f)
  pickle.dump(dataFrame, df_f)
```
Under else block:
```
with open(model_path.absolute(), "rb") as model_f, open(X_path.absolute(), "rb") as x_f, open(vectorizer_path.absolute(), "rb") as vector_f, open(dataframe_path.absolute(), "rb") as df_f:
  model = pickle.load(model_f)
  X = pickle.load(x_f)
  vectorizer = pickle.load(vector_f)
  dataFrame = pickle.load(df_f)
```
### Step 3:
Once the dataframe is ready, during the development, I have splitted the data into train and test and checked the accuracy of the model. Thats where I got the following accuracy - 
![image](https://user-images.githubusercontent.com/102677891/234177510-332533b2-b01f-44f5-ad73-5889e75a8ea3.png)
Upon the development stage is done, I went with production, that's where I have used the complete data to fit the model instead of test data.<br>
*Development stage code*
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.apply(lambda x: ' '.join(x))
X_test = X_test.apply(lambda x: ' '.join(x))
------------------------------------------------
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```
*Production stage code* - The considered code
```
X = X.apply(lambda x: ' '.join(x))
------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
model = LogisticRegression(random_state = 42, max_iter = 1000)
model.fit(X, y)
```


