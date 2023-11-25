import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Sentiment Recommendation System: ")

user_final_rating = pickle.load(open("user_final_rating.pkl",'rb'))
df = pd.read_csv("df.csv")
word_vectorizer = pickle.load(open("word_vectorizer.pkl",'rb'))
logit = pickle.load(open("logit_model.pkl","rb"))


def Recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    i = 0
    recomm = {}
    for prod_name in d.index.tolist():
        product_name=prod_name
        product_name_review_list = df[df['prod_name']==product_name]['Reviews'].tolist()
        features = word_vectorizer.transform(product_name_review_list)
        recomm[product_name] = logit.predict(features).mean()*100

    recommendation = pd.Series(recomm).sort_values(ascending=False).head(5).index.tolist()

    return  recommendation

def predict():

    username = st.text_input("Enter the user name: ")
    if st.button("Enter"):
        st.info(f"Recommendation for User : {username}")
        prediction = Recommend(username)
        st.write(f"Recommended Items:")
        st.table(prediction,)

predict()
