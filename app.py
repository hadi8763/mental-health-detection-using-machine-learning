from flask import Flask, request, render_template
import joblib
import nltk
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Manually set NLTK data path (change if needed)
nltk.data.path.append("C:/Users/User/AppData/Roaming/nltk_data")

# Download required NLTK data (safe to leave)
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # Avoids punkt tokenizer bug
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Flask setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    processed = preprocess_text(input_text)
    vectorized = vectorizer.transform([processed]).toarray()  # Convert sparse to dense for SVC
    prediction = model.predict(vectorized)[0]

    # Optional: Convert numerical label to friendly message
    label = "Needs Attention" if prediction == 1 else "No Mental Health Concern"
    return render_template('index.html', prediction=f"Prediction: {label}")

if __name__ == '__main__':
    app.run(debug=True)
