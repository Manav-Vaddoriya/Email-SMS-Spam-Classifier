from data_ingestion import load_data

def test_load_data_not_empty():
    df = load_data("experiments/spam.csv")
    assert df.shape[0] > 0
