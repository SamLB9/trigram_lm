from src.preprocessing import preprocess_line

def test_add_period_and_hash():
    assert preprocess_line("Hello") == "hello.#"

def test_remove_punct_and_digit():
    out = preprocess_line("Foo! Bar 123")
    assert out == "foo bar 000.#"