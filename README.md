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
