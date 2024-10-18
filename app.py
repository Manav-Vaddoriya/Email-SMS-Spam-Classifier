import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message: ")
if st.button('Predict'):
    def transform_text(text):
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        text = nltk.word_tokenize(text)
        
        y = []
        for i in text:
            if i.isalnum():  # Only keep alphanumeric words
                y.append(i)
        
        text = y[:]
        y.clear()
        
        # Remove stopwords and punctuation
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
        text = y[:]
        y.clear()
        
        # Apply stemming
        for i in text:
            y.append(ps.stem(i))
        
        return " ".join(y)

    # 1. Preprocess the input
    transform_sms = transform_text(input_sms)

    # 2. Vectorize the input
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict using the model
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
