## cs5293sp23-project2
### Project2 - Text Analytics - Cuisine Prediction
### Vasu Deva Sai Nadha Reddy Janapala
## project2.py file explaination
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

### Step 1:
In the main method, I have included two parameters - N and ingredient, which are the command line arguments. These inputs are expected from the user using the following command - 
```
pipenv run python project2.py --N 5 --ingredient "dry white wine" --ingredient "olive oil" --ingredient "garlic cloves" --ingredient lemon
```
After meeting the requirement, we move on to **step 2**.

### Step 2:
Now, the arguments are passed to the *cuisin_predict()* method. After passing the arguments, I parsed the information to local variables **(n, input_ingredients)**.
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
Once the dataframe is ready, during the development, I have splitted the data into train and test and checked the accuracy of the model. Thats where I got the following accuracy - <br>
![image](https://user-images.githubusercontent.com/102677891/234177510-332533b2-b01f-44f5-ad73-5889e75a8ea3.png)<br>
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
### Step 4:
Predicting the cuisine based on the input ingredients - *input_ingredients* along with that we find the score of the predicted cuisine. <br>
Post that we find the *cosine similarity*. Which is used to find the nearest cuisines. <br>
The values are stored respectively and prints also returns the content to the main function to push that data into the json file, as shown below - <br>
```
list1 = []
count = 0
length = len(nearest_cuisines_ids)
while(count < length):
    list1.append({'id': nearest_cuisines_ids[count], 'score': nearest_cuisines_scores[count]})
    count += 1
output_dict = {
    "cuisine": predicted_cuisine,
    "score": predicted_cuisine_score,
    "closest": list1
}
json_str = json.dumps(output_dict, indent=2)
print(json_str)
return output_dict
```
### Step 5 - Output <br>
![image](https://user-images.githubusercontent.com/102677891/234186819-2c839cd5-2a95-40f7-9f23-5e7bf9826506.png) <br>
Under the docs folder, I have saved a screen recorded video, to show the successful run of the code.

*In the entire project, I have used the yummly.json as the inputfile*


## test_project2.py explaination
In the pytest, I have Arranged, Act and Asserted the data as follows
```
def test_cuisin_predict():

# creating the N and ingredient values
args = argparse.Namespace(N = "5", ingredient = ["white bread", "white onion", "grape tomatoes", "vegetable oil"])

# Call the function with the test arguments
output_dict = cuisin_predict(args)

# Assert that the output dictionary has the expected keys and values
assert set(output_dict.keys()) == {'cuisine', 'score', 'closest'}
# breakpoint()
assert isinstance(output_dict['cuisine'], str)
assert isinstance(output_dict['score'], float)
assert isinstance(output_dict['closest'], list)
assert all(isinstance(d, dict) for d in output_dict['closest'])
assert all(set(d.keys()) == {'id', 'score'} for d in output_dict['closest'])
assert all(isinstance(d['id'], str) for d in output_dict['closest'])
assert all(isinstance(d['score'], float) for d in output_dict['closest'])
```
Used the following command line
```
pipenv run python -m pytest
```
The below is the pytest result - <br>
![image](https://user-images.githubusercontent.com/102677891/234191392-30cbbeae-a56f-4365-afe8-d80873e983bb.png)
The successful result from the pytest, for which I used the breakpoint and displayed the result.<br>
![image](https://user-images.githubusercontent.com/102677891/234191136-f741578d-0933-4f1e-ba63-a1689d7c82f1.png)
