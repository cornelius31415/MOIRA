#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:08:52 2024

@author: cornelius
"""

# ------------------------         IMPORTS         ---------------------------


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("commands.csv")


# ------------------------ BAG OF WORDS DATAFRAME  ---------------------------


sentences = df['Command'].values.tolist()
actions = df['Action'].values.tolist()

count_vec = CountVectorizer()
word_counts = count_vec.fit_transform(sentences)
bag_of_words_df = pd.DataFrame(word_counts.toarray(),columns = count_vec.get_feature_names_out())

num_features = len(count_vec.get_feature_names_out())
num_actions = len(set(actions))



label_encoder = LabelEncoder()
df["Action"] = label_encoder.fit_transform(df["Action"])


# ---------------------------     NEURAL NETWORK    ----------------------------

neuralNetwork = NeuralNetwork(num_features,1e-2)
neuralNetwork.layer(50)
neuralNetwork.layer(num_actions)
neuralNetwork.fit(bag_of_words_df,df['Action'],100)




# ---------------------------    DECISION TREE    ----------------------------



decision_tree = DecisionTreeClassifier()
decision_tree.fit(bag_of_words_df.values, df['Action'].values)



# -----------------------------    CHATBOT     -------------------------------


print()
print()
print("---------------------------------------------------------------")
print("----------------     WELCOME TO CHATTY      -------------------")
print("---------------------------------------------------------------")
print("\n")

while True:

    user_input = input("Your Message: ")
    print()
    
    prediction_index = neuralNetwork.predict(count_vec.transform([user_input]))[0]
    prediction_label = label_encoder.inverse_transform([prediction_index])[0]
    print(prediction_label)
    
    print()