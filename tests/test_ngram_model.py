from src.ngram_model import CharTrigramModel

def test_train_and_generate():
    text = "##abc.# abc.#"
    model = CharTrigramModel(alpha=0.5)
    model.train_from_text(text)
    gen = model.generate_text(20)
    # Should at least return something and never crash
    assert isinstance(gen, str) and len(gen) == 20