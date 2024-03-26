from flask import Flask, render_template, request
import pickle
from snownlp import SnowNLP

app = Flask(__name__)

# Load the SnowNLP model from the pickle file
with open("s.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        s = SnowNLP(text)
        keywords = s.keywords(5)
        summary = s.summary(2)  # Limiting to 2 sentences for summary
        sentences = s.sentences
        sentiments = [SnowNLP(sentence).sentiments for sentence in sentences]
        return render_template('index.html', text=text, keywords=keywords, summary=summary, sentences=sentences, sentiments=sentiments)
        
if __name__ == '__main__':
    app.run(debug=True)
