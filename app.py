from flask import Flask, render_template, request
import pickle
import re
import string
import nltk

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

# Load model and vectorizer
model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return render_template('index.html', prediction=prediction, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
