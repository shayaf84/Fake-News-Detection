<b> Folders: </b>

<i> Models: </i>

1. Dense Layer Model - Gensim Doc2Vec Word Vectors
2. LSTM Model - Word2Vec Vectors
3. BERT Model


<i> Templates: </i>

1. index.html - Front end of the model -> programmed with html and css
2. sub.html - Redirect after submission


<b> Files: </b>

app.py - Back end programmed in Flask


<b> One will need to download all the models in the Models folder for use in app.py. File sizes are large as SOTA models have larger parameters </b>

<b> Implementation </b>

Download all folders and set up your environment (I used colab). Run all files in the "Models" folder and download the pickle file that denotes each of the models.
From there, use this code to connect to an ngrok terminal:

```
!pip3 install tensorflow_text==2.7.0
!pip3 install flask-ngrok

! curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok   

! ngrok authtoken [PUBLIC KEY]

```
Ensure all the models, templates, and the app.py file are in the same directory.

Then, you can run the app.py file

<img width="1440" alt="Screen Shot 2022-02-07 at 10 45 57 PM" src="https://user-images.githubusercontent.com/68475848/158228232-7d75b872-6d45-43df-a0f2-748968791309.png">
