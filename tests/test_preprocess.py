from src.preprocess import clean_text

def test_clean():
    txt = "<b>Hello!!!</b>"
    assert clean_text(txt) == "hello"
