from model_building import train_model
from sklearn.feature_extraction.text import TfidfVectorizer

def test_model_training():
    texts = ["free prize", "hello friend"]
    labels = [1, 0]

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)

    model = train_model(X, labels)
    assert hasattr(model, "predict")
