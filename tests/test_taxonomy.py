import numpy as np
from src import embed, taxonomy


def test_embed_returns_normalised_vectors():
    v = embed.embed(["flight cancelled", "lost my bag"])
    assert v.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_embed_is_deterministic():
    a = embed.embed(["flight cancelled"])
    b = embed.embed(["flight cancelled"])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_similar_texts_are_closer_than_dissimilar_ones():
    v = embed.embed(["my flight was cancelled",
                     "my flight got cancelled today",
                     "how do I join the Executive Club"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_cluster_separates_two_obvious_groups():
    v = embed.embed(["lost baggage", "my bag is missing", "baggage never arrived",
                     "flight delayed", "flight is late", "delayed departure"])
    labels = taxonomy.cluster(v, k=2)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_top_terms_surfaces_discriminative_words():
    texts = ["lost baggage", "missing baggage", "flight delayed", "flight late"]
    labels = np.array([0, 0, 1, 1])
    terms = taxonomy.top_terms(texts, labels, k=3)
    assert "baggage" in terms[0]
    assert "flight" in terms[1] or "delayed" in terms[1]


def test_load_intents_includes_other_and_is_reasonably_small():
    intents = taxonomy.load_intents()
    assert "other" in intents
    assert 5 <= len(intents) <= 12
    assert len(intents) == len(set(intents))
