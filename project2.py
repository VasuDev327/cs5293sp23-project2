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

def cuisin_predict(argument):
    # reading the arguments into local variables
    n = int(argument.N[0])
    input_ingredients = argument.ingredient
    # print(n)
    # print(ingredients)
    # print(type(n))
    # raise

    # the below are the pickle paths to store the read information from the model
    model_path = Path.cwd() / "logistic.pkl"
    X_path = Path.cwd() / "X_tra.pkl"
    vectorizer_path = Path.cwd() / "vectorizer.pkl"
    dataframe_path = Path.cwd() / "dataframe.pkl"

    # if the pickle paths does not exists we go to through this if statement else it goes through the else statement
    if not model_path.exists() or not X_path.exists() or not vectorizer_path.exists():

        # open the json file
        f = open("yummly.json")

        # loads the json file
        data = json.load(f)

        # creating a dataframe
        dataFrame = pd.DataFrame()

        # loop to read the json data and insert it into the created data frame
        for d in data:
            dataFrame = dataFrame._append(d, ignore_index=True)
        
        # closing the file
        f.close()

        # reading the ingredients variable
        X = dataFrame['ingredients']

        # reading the label
        y = dataFrame['cuisine']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the list of ingredients to a string
        # X_train = X_train.apply(lambda x: ' '.join(x))
        # X_test = X_test.apply(lambda x: ' '.join(x))

        # Convert the list of ingredients to a string
        X = X.apply(lambda x: ' '.join(x))

        # Create a vectorizer to convert the ingredients into numerical features
        vectorizer = TfidfVectorizer()
        # X_train = vectorizer.fit_transform(X_train)
        # X_test = vectorizer.transform(X_test)
        X = vectorizer.fit_transform(X)

        # logistic regression model
        model = LogisticRegression(random_state = 42, max_iter = 1000)

        # fitting the model
        model.fit(X, y)

        # y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print('Accuracy:', accuracy)

        # dumping the preprocessed, trained data into the form of pickle for further use of this information
        with open(model_path.absolute(), "wb") as model_f, open(X_path.absolute(), "wb") as x_f, open(vectorizer_path.absolute(), "wb") as vector_f, open(dataframe_path.absolute(), "wb") as df_f:
            pickle.dump(model, model_f)
            pickle.dump(X, x_f)
            pickle.dump(vectorizer, vector_f)
            pickle.dump(dataFrame, df_f)
    else:
        # saved pickle formats are pulled here, if we overcome the if statement
        with open(model_path.absolute(), "rb") as model_f, open(X_path.absolute(), "rb") as x_f, open(vectorizer_path.absolute(), "rb") as vector_f, open(dataframe_path.absolute(), "rb") as df_f:
            model = pickle.load(model_f)
            X = pickle.load(x_f)
            vectorizer = pickle.load(vector_f)
            dataFrame = pickle.load(df_f)

    # the ingredients inputs which we received from the command line
    input_ingredients = ' '.join(input_ingredients)

    # transforming the input using the vectorization concept
    input_features = vectorizer.transform([input_ingredients])

    # using the model to predict the cuisine based on the inputs
    predicted_cuisine = model.predict(input_features)[0]

    # predicting the score of the found cuisine
    predicted_cuisine_index = list(model.classes_).index(predicted_cuisine)
    predicted_cuisine_score = model.predict_proba(input_features)[0][predicted_cuisine_index]

    # using the cosine similarity to check the similarity score
    similarity_scores = cosine_similarity(input_features, X)
    similarity_scores = similarity_scores[0]

    # findind the nearest cusinines based on the similary scores
    nearest_cuisines_indices = similarity_scores.argsort()[::-1][1:n+1]  # exclude the predicted cuisine itself
    nearest_cuisines = dataFrame.iloc[nearest_cuisines_indices]
    nearest_cuisines_ids = nearest_cuisines['id'].values
    nearest_cuisines_ids = [str(id) for id in nearest_cuisines_ids]
    nearest_cuisines_scores = similarity_scores[nearest_cuisines_indices]

    # Create a dictionary with the relevant information
    list1 = []
    count = 0
    length = len(nearest_cuisines_ids)
    while(count < length):
        list1.append({'id': nearest_cuisines_ids[count], 'score': round(nearest_cuisines_scores[count], 2)})
        count += 1
    output_dict = {
        "cuisine": predicted_cuisine,
        "score": round(predicted_cuisine_score, 2),
        "closest": list1
    }
    json_str = json.dumps(output_dict, indent=2)
    print(json_str)
    return output_dict

# main method to pass the arguments from the CLI (command line interface)
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Cuisine prediction')
    arg_parser.add_argument('--N', required=True, action="append", help='number of nearest cuisines')
    arg_parser.add_argument('--ingredient', required=True, action='append', help='ingredient names')
    args = arg_parser.parse_args()

    if args.N and args.ingredient:
        output = cuisin_predict(args)
        with open('output.json', 'w') as f:
            json.dump(output, f, indent=2)