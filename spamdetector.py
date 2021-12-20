
#spam filter from scratch

import pandas as pd
import numpy as np
import re

# importing data
data = pd.read_csv('dataset.csv', encoding="unicode_escape")

# display the first few entries
print(data.head())

data = data[['label', 'text']]
print(data.head())

print(type(data))

# to display the dataset details
print(data.describe())

# to display the spam and ham count
print(data.label.value_counts())

# Data cleaning using regular expression to match words only

def clean_data(email):
    return " ".join(re.findall(r"\b[a-zA-Z]+\b(?<!subject)", email.lower()))

data['text'] = data['text'].apply(lambda x: clean_data(x))
data.head()

from sklearn.model_selection import train_test_split

X, Y = data['text'], data['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15,random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

vectorized_data = vectorizer.fit_transform(x for x in X_train)

vectorized_data = pd.DataFrame(vectorized_data.toarray())
vectorized_data.head()


# Setting the column names as word tokens

tfidf_tokens = vectorizer.get_feature_names()
vectorized_data = vectorized_data.set_axis(tfidf_tokens, axis=1, inplace=False)
vectorized_data.head()

# Appending label to the corresponding vectors

vectorized_data['label'] = data['label']
vectorized_data.head()

# Summing up the likelihood of each token

p_dist = vectorized_data.groupby('label').sum()
p_dist.head()

# adding to token to avoid multiplication with '0'

p_dist += 1
p_dist.head()

# Normalizing the values between 0 and 1 by dividing all the values by max(all the values)

p_dist.loc['ham'] = p_dist.loc['ham'] / p_dist.max(axis=1)[0]
p_dist.loc['spam'] = p_dist.loc['spam'] / p_dist.max(axis=1)[1]

# Display normalized values

p_dist.head()

p_ham = (data['label'] == 'ham').sum() / data.shape[0]
p_spam = (data['label'] == 'spam').sum() / data.shape[0]

print(p_ham, p_spam)

# Defining Naive Bayes function to calculate the chance of a given input text being spam and ham

def naive_bayes(p_dist, email, p_ham, p_spam):
    tokens = re.findall(r"\w[a-zA-Z]+", email)
    ham_prob, spam_prob = p_ham, p_spam
    for token in tokens:
        if token in p_dist:
            ham_prob = ham_prob * p_dist[token][0]
            spam_prob = spam_prob * p_dist[token][1]

    return ham_prob, spam_prob

test_set = pd.DataFrame([X_test, Y_test]).transpose()
test_set.head()

def prediction_accuracy(p_dist, test_set, p_ham, p_spam):
    predicted_correct = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for index, row in test_set.iterrows():
        ham_score, spam_score = naive_bayes(p_dist, row['text'], p_ham, p_spam)
        if (spam_score > ham_score):
            if row['label'] == 'spam':
                TP += 1
                predicted_correct += 1
            else:
                FP += 1
        else:
            if row['label'] == 'ham':
                TN += 1
                predicted_correct += 1
            else:
                FN += 1

    accuracy = (predicted_correct / test_set.shape[0]) * 100
    return accuracy, TP, TN, FP, FN

prediction_results = prediction_accuracy(p_dist, test_set, p_ham, p_spam)
print(f'Accuracy: {prediction_results[0]:.2f}%')

print("Confusion Matrix")
print('         Positive   Negative')
print(f'Positive {prediction_results[1]}        {prediction_results[3]}')
print(f'Negative {prediction_results[4]}         {prediction_results[2]}')


