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

#Loading Model and Vectorizer
sentiment_analyzer = pickle.load(open('/home/aritrabag/Documents/internship_projects/setiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('/home/aritrabag/Documents/internship_projects/vectorizer.pkl', 'rb'))

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
    #print(output)
    #print(negativeness)
    #print(positiveness)
    return (str(positiveness), str(negativeness), output)
    
#predict_sentiment('True that. Absolutely bad bad movie, would not rate it more than 5/10. @Theodor #HNY')
#predict_sentiment('Great Day. @R@me$h')


with gr.Blocks() as Post_Sentiment_Analyzer:
    post = gr.Textbox(label = "Enter Tweet / Post")
    positiveness = gr.Textbox(label =" Positive Feeling (%) ")
    negativeness = gr.Textbox(label =" Negative Feeling (%) ")
    output = gr.Textbox(label =" Overall Sentiment of the Tweet ")
    prediction_btn = gr.Button("Predict Sentiment")
    prediction_btn.click(fn = predict_sentiment, inputs = post, outputs= [positiveness, negativeness, output], api_name="tweet_sentiment_analyzer")
    
Post_Sentiment_Analyzer.launch(share = True)  
