from model_evaluation import evaluate_model

def test_accuracy_range():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    acc = evaluate_model(y_true, y_pred)
    assert 0 <= acc <= 1
