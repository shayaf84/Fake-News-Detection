from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import ssl
import tensorflow_text as text
import joblib
from flask_ngrok import run_with_ngrok


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#Initializing the Flask app and running with ngrok
app = Flask(__name__)
run_with_ngrok(app)

#Uploading the models and storing in variables
model = pickle.load(open('model.pkl','rb'))
lstm = pickle.load(open('lstm (2).pkl','rb'))
bert = joblib.load('bert.pkl')
doc = pickle.load(open('doc.pkl','rb'))
word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

@app.route("/")

# Displaying front end
def hello():
    return render_template('index.html')


@app.route('/sub', methods=["POST"])

#Retrieving submission with 'POST' and storing the name and model_type
def submit():
    if request.method == 'POST':
        name = request.form['news']
        modelType = request.form['modelType']
    
    #New sentence for the model
    sentence = name

    #Lowercase words
    sentence = sentence.lower()

    #Ensure that all necessary punctuations are in one list
    #Include ' and " as they are not default
    punc = list(string.punctuation)
    punc.append('\'')
    punc.append('"')
    print(punc)

    
    #Loop through sentence and remove all punctuations
    for i in string.punctuation:
        sentence = sentence.replace(i, '')
    
    #Tokenize sentence -> all words in a new list
    tok = sentence.split(' ')

    #Define text lemmatization model (eg: walks will be changed to walk)
    lemmatizer = WordNetLemmatizer()

    #Lemmatize each word in the sentence
    for w in sentence:
        lemmatizer.lemmatize(sentence)

    #Define all stopwords in the English language (it, was, for, etc.)
    stop = stopwords.words('english')

    #Remove them from our dataframe and store in a new list
    minStop = []

    for i in tok:
        if i not in stop:
            minStop.append(i)

    #If the model is a dense network, infer Doc2Vec predictions and pass the vectors into the dense model. Store results in a variable
    if modelType == 'dense':
        #Doc2Vec tags
        tag = [TaggedDocument(minStop,[0])]

        predVec = [doc.infer_vector(minStop)]
        predVec = np.array(predVec)
        
        

        results = model.predict(predVec)
    #If the model is an LSTM network, infer Word2Vec predictions and pass the vectors into the lstm model. Store results in a variable
    elif modelType == 'lstm':
        
        val = []

        
        for i in minStop:
            temp = np.array(word2vec([i]))
            val.append(temp)

        val = np.array(val)
        results = lstm.predict(val)
    #If the model is a BERT model, pass the raw text into the BERT pipeline, as preprocessing is included there. Store results in a variable
    elif modelType == 'bert':
       
        results = bert.predict([name])

        


    #Index into results. If probability is greater than 0.5, fake news. Else, real news.
    conVal = results[0][0]
    
    if results[0][0] >= 0.5:
        results = "Fake"
        
    else:
        results = "True"
    #Convert probability to percentages and display on the front-end
    conVal *=100 
    conVal = round(conVal,3)
    conVal = str(conVal) + ' %'
    return render_template('index.html', official = results,con=conVal)




if __name__ == "__main__":
    app.run()
