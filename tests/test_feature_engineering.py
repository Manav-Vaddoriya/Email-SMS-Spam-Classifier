from feature_engineering import build_vectorizer

def test_vectorizer_vocab_created():
    texts = ["free money", "hello friend"]
    vectorizer = build_vectorizer(texts)
    assert len(vectorizer.vocabulary_) > 0
