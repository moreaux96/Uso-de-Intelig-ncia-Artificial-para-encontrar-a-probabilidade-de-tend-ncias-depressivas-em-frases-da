#cleansing data

import pandas as pd
import string
import nltk


dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
dataset_encode = "ISO-8859-1"
data = pd.read_csv("dataset_random", encoding = dataset_encode, names = dataset_columns)

data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

data['text'].isnull().sum()

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
data['clean_text']=data['text'].apply(lambda x: remove_punctuation(x))
data.head()

data['clean_text'] = data['clean_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
data['clean_text'] = data['clean_text'].str.lower()

nltk.download('punkt')

def tokenize(text):
    split=re.split("\W+",text) 
    return split
data['clean_text_tokenize']=data['clean_text'].apply(lambda x: tokenize(x.lower()))


nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('portuguese')

def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
data['clean_text_tokenize_stopwords'] = data['clean_text_tokenize'].apply(lambda x: remove_stopwords(x))
data.head(10)
