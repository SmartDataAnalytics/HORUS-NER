from config import HorusConfig
from src.algorithms.text_classification.bow_tfidf import BowTfidf

config = HorusConfig()

def test_models_text_v1():
    model = BowTfidf(config)
    out = model.detect_text_klass('diego esteves is a nice guy that lives in Paris')
    print(out)


if __name__ == '__main__':
    test_models_text_v1()

