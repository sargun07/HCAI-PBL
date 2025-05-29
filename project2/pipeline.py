import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class IMDBDataLoader:
    def __init__(self, filedirectory, filename):
        self.filedirectory = filedirectory
        self.filepath = os.path.join(filedirectory, filename)
        self.df = None

    def load_data(self):
        import pandas as pd
        self.df = pd.read_csv(self.filepath)
        return self.df

    def get_wordnet_pos(self, treebank_tag):
      
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  

    def preprocess(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        preprocessed_texts = []

        for text in self.df["review"]:
            
            text = text.lower()

            text = re.sub(r'[^a-z\s]', '', text)

            tokens = word_tokenize(text)

            tagged_tokens = pos_tag(tokens)

            cleaned_tokens = [
                lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
                for word, pos in tagged_tokens
                if word not in stop_words
            ]

            cleaned_text = " ".join(cleaned_tokens)
            preprocessed_texts.append(cleaned_text)

        self.df["clean_review"] = preprocessed_texts


    def save_preprocessed_data(self, output_filename="imdb_dataset_preprocessed.csv"):
        output_path = os.path.join(os.path.dirname(self.filedirectory), output_filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        self.df.to_csv(output_path, index=False)


class SVMTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LinearSVC()
        self.label_encoder = LabelEncoder()
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def train(self, df):

        y = self.label_encoder.fit_transform(df['sentiment'])
        X = self.vectorizer.fit_transform(df['clean_review'])

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
    
    def evaluate(self, model=None, X_test=None, y_test=None):
        if model is None:
            model = self.model
        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test

        if model is None or X_test is None or y_test is None:
            raise ValueError("Model and test data must be provided or trained before evaluation.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        matrix = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": matrix,
        }

    def save_model(self, path='Media/processed/project2/svm_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder
            }, f)

        
