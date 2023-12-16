# CODINGSAMURAI
This repository contains the files for the project of Sentiment Analysis. This internship was conducted virtually between Nov-17 and Dec-17, 2023

## Objective
To develop a model that can accurately predict the positive and negative sentiments expressed in tweets/posts

## Data Description
Source: https://www.kaggle.com/code/aritrabag/twitter-sentiment-analysis/input
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the Twitter API. The tweets have been annotated (0 = negative, 4 = positive)
It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)

## Working data Size

The entire dataset is too big to work with considering the hardware and time limitations of Kaggle and the requirements of hyperparameter tuning. Hence half of it was used. This includes the first 400,000 tweets with positive sentiments and the first 400,000 tweets with negative sentiments, for a total of 800,000 tweets. They are equally distributed to avoid any biases in the final training and test sets. 85% of this working data was used for training and the remaining 15% for testing purposes.

## Data Preprocessing Steps

1) stopword removal (Using the NLTK English Stopword List)
   
2) converting to lower case

3) url removal

4) punction removal

5) repeated character removal

6) numerical character removal

7) removing user@ references and #

8) tokenization (Using RegexpTokenizer)

9) stemming (Using PorterStemmer)

10) lemmatizing (Using WordNetLemmatizer)

Further Vectorization was carried out with TfidfVectorizer


##  Post Processing Exploratory Data Analysis Results

Sentiment Data Distribution
1 : Positive
0 : Negative

![Screenshot from 2023-12-16 12-05-08](https://github.com/riktiger/CODINGSAMURAI/assets/57993082/7e04f16c-f9e1-4e64-9f02-98322254171d)

Words most frequently used in Positive Sentiment tweets

![positive_word](https://github.com/riktiger/CODINGSAMURAI/assets/57993082/1d056e1c-0c1b-487d-b314-9fe35238caa8)

Words most frequently used in Negative Sentiment tweets

![negative_word](https://github.com/riktiger/CODINGSAMURAI/assets/57993082/4aacda7f-08c0-4c91-8625-ea5b4bee03b7)


## Hyper-parameter Tuning Results

Model Used: Logistic Regression

Best Score: 72.02833333333334

Best Random State: 17 (after 5 iterations)


## Model Evaluation on Test Data

![Screenshot from 2023-12-16 12-04-52](https://github.com/riktiger/CODINGSAMURAI/assets/57993082/40ff8a48-6e75-4466-9fd0-3f8c3ed75840)


## Challenges Faced
The biggest challenge in this entire project would be the collection of data. As of December 2023, Twitter has officially blocked access to the API, hence the source for sentiment analysis, i.e. Sentiment140 does not contain the information. Other sources of data are too biased towards a certain topic, like the  US presidential election campaign of 2016, hence getting a relatively neutral data source, especially a labeled one was difficult
The only data source capable of being the basis of the sentiment-predicting model is given at https://www.kaggle.com/code/aritrabag/twitter-sentiment-analysis/input. However, it does not contain any details about the tweets that are neutral. Manually labeling  outputs that are 50% positive and 50% negative as neutral will be unfair to the data and the neutrality of the model and hence that aspect of the project was left out on purpose.
In the case of the adaptive model, the training dataset used to retrain the model once a user inputs a sentiment of his choice is small. Hence the biasness of words is high. In a practical situation, the entire training dataset along with the new data should be used to retrain the model. Training requires an extensive amount of time considering the available hardware. 
