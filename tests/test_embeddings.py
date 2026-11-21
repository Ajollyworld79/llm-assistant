import math
from dataclasses import replace

import pytest

from backend.app.services.embeddings import Embedder, EmbeddingError, hash_embedding
from tests.conftest import DIM, FakeEmbeddings, fake_gemini


def test_hash_embedding_is_deterministic_unit_vector():
    a = hash_embedding("hello", 64)
    assert a == hash_embedding("hello", 64) and a != hash_embedding("other", 64)
    assert len(a) == 64 and math.isclose(sum(x * x for x in a), 1.0)


async def test_demo_uses_hash_vectors(settings):
    vecs = await Embedder(settings, None).embed(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == DIM
    assert await Embedder(settings, None).embed([]) == []


async def test_non_demo_without_client_raises_instead_of_faking(prod_settings):
    embedder = Embedder(prod_settings, None)
    assert not embedder.ready
    with pytest.raises(EmbeddingError):
        await embedder.embed(["text"])


async def test_provider_failure_raises_not_falls_back(prod_settings):
    client = fake_gemini()
    client.aio.models.fail = True
    with pytest.raises(EmbeddingError):
        await Embedder(prod_settings, client).embed(["text"])


async def test_gemini_success_sets_task_type_and_normalises(prod_settings):
    client = fake_gemini()
    embedder = Embedder(prod_settings, client)
    doc = await embedder.embed(["a", "b"])
    query = await embedder.embed(["q"], is_query=True)
    assert len(doc) == 2 and len(query[0]) == DIM
    calls = client.aio.models.calls
    assert calls[0][2] == "RETRIEVAL_DOCUMENT" and calls[1][2] == "RETRIEVAL_QUERY"


async def test_wrong_dimension_is_rejected(prod_settings):
    client = fake_gemini(dim=DIM + 1)
    with pytest.raises(EmbeddingError, match="dimensions"):
        await Embedder(prod_settings, client).embed(["x"])


async def test_batches_large_inputs(prod_settings):
    client = fake_gemini()
    out = await Embedder(prod_settings, client).embed([str(i) for i in range(250)])
    assert len(out) == 250 and [c[1] for c in client.aio.models.calls] == [100, 100, 50]


async def test_count_mismatch_is_rejected(prod_settings):
    class Short(FakeEmbeddings):
        async def embed_content(self, **kwargs):
            res = await super().embed_content(**kwargs)
            res.embeddings.pop()
            return res

    client = fake_gemini()
    client.aio.models = Short(DIM)
    with pytest.raises(EmbeddingError, match="unexpected number"):
        await Embedder(prod_settings, client).embed(["a", "b"])


async def test_timeout_is_an_embedding_error(prod_settings):
    import asyncio

    class Slow(FakeEmbeddings):
        async def embed_content(self, **kwargs):
            await asyncio.sleep(5)

    client = fake_gemini()
    client.aio.models = Slow(DIM)
    with pytest.raises(EmbeddingError):
        await Embedder(replace(prod_settings, embedding_timeout=0), client).embed(["a"])
