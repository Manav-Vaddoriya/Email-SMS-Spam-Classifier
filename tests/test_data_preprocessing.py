from data_preprocessing import clean_text

def test_clean_text_lowercase():
    assert clean_text("FREE MONEY") == "free money"

def test_clean_text_removes_special_chars():
    assert "!" not in clean_text("win!!!")
