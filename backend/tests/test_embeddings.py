import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import embeddings


def test_normalize_list_of_vectors():
    out = [[0.1, 0.2], [0.3, 0.4]]
    res = embeddings._normalize_vectors(out)
    assert isinstance(res, list)
    assert len(res) == 2


def test_normalize_single_vector():
    out = [0.1, 0.2, 0.3]
    res = embeddings._normalize_vectors(out)
    assert res == [[0.1, 0.2, 0.3]]


def test_normalize_dict_embeddings():
    out = {'embeddings': [[0.5, 0.5], [0.1, 0.9]]}
    res = embeddings._normalize_vectors(out)
    assert len(res) == 2


def test_normalize_data_items():
    out = {'data': [{'embedding': [0.1, 0.2]}, {'embedding': [0.3, 0.4]}]}
    res = embeddings._normalize_vectors(out)
    assert len(res) == 2


def test_normalize_api_like_response():
    out = {'data': [{'embedding': [0.1, 0.2]}, {'embedding': [0.3, 0.4]}], 'other': 'meta'}
    res = embeddings._normalize_vectors(out)
    assert len(res) == 2
