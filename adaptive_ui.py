# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing Libraries
import gradio as gr
import numpy as np
import sklearn
import nltk
import re
import string
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Importing stopwords list
##nltk.download('punkt')
##nltk.download('omw-1.4')
##nltk.download('wordnet')
##nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Loading Vectorizer and Default Model
vectorizer = pickle.load(open('/home/aritrabag/Documents/internship_projects/adaptive_model/adaptive_vectorizer.pkl', 'rb'))
default_model = pickle.load(open('/home/aritrabag/Documents/internship_projects/adaptive_model/default_setiment_model.pkl', 'rb'))

#Tweet Preprocessing Function Creation
def preprocess_data (input):
    
    #stopword removal
    input = " ".join([word for word in str(input).split() if word not in stop_words])
    
    #converting to lower case
    input = input.lower()
    
    #url removal
    input = re.sub(r"http\S+|www\S+|https\S+","",input,flags = re.MULTILINE)
    
    #punction removal
    input = input.translate(str.maketrans("", "", string.punctuation))
                        
    #repeated character removal
    input = re.sub(r'(.)\1+', r'\1', input)
                        
    #numerical character removal
    input = re.sub('[0-9]+', '', input)
    
    #removing user@ references and #
    input = re.sub(r'\@\w+|\#w+', "", input)
                     
    #tokenization
    tokens = RegexpTokenizer(r'\w+')
    filtered_list = tokens.tokenize(input)
    
    #stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in filtered_list]
      
    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos = 'a') for w in stemmed_words]
    
    #output
    return " ".join(lemma_words)

##input = "True that. Absolutely bad bad movie, would not rate it more than 5/10. @Theodor #HNY"
##preprocess_data (input)

def predict_sentiment(input):
    sentiment_analyzer = pickle.load(open('/home/aritrabag/Documents/internship_projects/adaptive_model/adaptive_setiment_model.pkl', 'rb'))
    processed_input = preprocess_data(input)
    processed_input = [processed_input]
    vectorized_data = vectorizer.transform(processed_input)
    arrayed_input = vectorized_data.toarray()
    negativeness = round((float(sentiment_analyzer.predict_proba(arrayed_input)[0][0])*100),2)
    positiveness = round((float(sentiment_analyzer.predict_proba(arrayed_input)[0][1])*100),2)
    sentiment = int(sentiment_analyzer.predict(arrayed_input)[0])
    if sentiment == 0:
        output = 'Negative'
    elif sentiment == 1:
        output = 'Positive'
    if positiveness == 50.0:
        output = 'Neutral'
    #print(output)
    #print(negativeness)
    #print(positiveness)
    return (str(positiveness), str(negativeness), output)
    
#predict_sentiment('True that. Absolutely bad bad movie, would not rate it more than 5/10. @Theodor #HNY')
#predict_sentiment('Great Day. @R@me$h')


def model_feedback(input, user_defined_sentiment):
    sentiment_analyzer = pickle.load(open('/home/aritrabag/Documents/internship_projects/adaptive_model/adaptive_setiment_model.pkl', 'rb'))
    if user_defined_sentiment == 'Positive' or 'positive':
        data = [['bad' , 0], [input, 1]]
        df = pd.DataFrame(data, columns=['posts', 'sentiment'])
        df['posts'] = df['posts'].apply(lambda x : preprocess_data(x))
        X = df['posts']
        Y = df['sentiment']
        X = vectorizer.transform(X)
        X = X.toarray()
        sentiment_analyzer.fit(X,Y)
    elif user_defined_sentiment == 'Negative' or 'negative':
        data = [['great' , 1], [input, 0]]
        df = pd.DataFrame(data, columns=['posts', 'sentiment'])
        df['posts'] = df['posts'].apply(lambda x : preprocess_data(x))
        X = df['posts']
        Y = df['sentiment']
        X = vectorizer.transform(X)
        X = X.toarray()
        sentiment_analyzer.fit(X,Y)
    
    pickle.dump(sentiment_analyzer, open('/home/aritrabag/Documents/internship_projects/adaptive_model/adaptive_setiment_model.pkl','wb'))
    

def reset():
    pickle.dump(default_model, open('/home/aritrabag/Documents/internship_projects/adaptive_model/adaptive_setiment_model.pkl','wb'))
    
    
with gr.Blocks() as Post_Sentiment_Analyzer:
    post = gr.Textbox(label = "Enter Tweet / Post")
    user_defined_sentiment = gr.Textbox(label = " User defined sentiment of the Tweet / Post as Positive or Negative (For Feedback) ")
    positiveness = gr.Textbox(label =" Detected Positive Feeling (%) ")
    negativeness = gr.Textbox(label =" Detected Negative Feeling (%) ")
    output = gr.Textbox(label =" Detected Overall Sentiment of the Tweet ")
    prediction_btn = gr.Button("Predict Sentiment")
    prediction_btn.click(fn = predict_sentiment, inputs = post, outputs= [positiveness, negativeness, output])
    feedback_btn = gr.Button("Feedback")
    feedback_btn.click(fn = model_feedback, inputs = [post,user_defined_sentiment])
    reset_btn = gr.Button("Reset System to Defaults")
    reset_btn.click(fn = reset)
    
Post_Sentiment_Analyzer.launch(share = True)  
