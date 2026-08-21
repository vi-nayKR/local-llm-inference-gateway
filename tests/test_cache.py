import unittest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gateway.cache import LightweightEmbedder, RedisSemanticCache

class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self.embedder = LightweightEmbedder(dimension=384)
        self.cache = RedisSemanticCache(threshold=0.60)

    def test_embedding_dimension_and_norm(self):
        vec = self.embedder.encode("What is retrieval augmented generation?")
        self.assertEqual(len(vec), 384)
        
        # Norm should be close to 1.0 (unit vector)
        norm = sum(x * x for x in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_semantic_similarity(self):
        v1 = self.embedder.encode("How does vLLM paged attention work?")
        v2 = self.embedder.encode("Explain vLLM paged attention mechanism")
        v3 = self.embedder.encode("What is the recipe for chocolate cake?")
        
        sim_related = self.embedder.cosine_similarity(v1, v2)
        sim_unrelated = self.embedder.cosine_similarity(v1, v3)
        
        # Related questions should have substantially higher cosine similarity
        self.assertGreater(sim_related, sim_unrelated)
        self.assertGreater(sim_related, 0.55)
        self.assertLess(sim_unrelated, 0.20)

    def test_exact_cache_hit(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        hit = loop.run_until_complete(
            self.cache.get_cached_response("What is RAG in AI?")
        )
        self.assertIsNotNone(hit)
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(hit["similarity"], 1.0)
        self.assertLess(hit["latency_ms"], 5.0)  # Sub-5ms guarantee
        loop.close()

    def test_semantic_cache_hit_and_miss(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 1. New prompt should be cache miss initially
        miss = loop.run_until_complete(
            self.cache.get_cached_response("Explain quantum computing qubit superposition")
        )
        self.assertIsNone(miss)
        
        # 2. Store response
        loop.run_until_complete(
            self.cache.set_cached_response(
                "Explain quantum computing qubit superposition",
                "Superposition is a principle of quantum mechanics where a qubit exists in multiple states simultaneously."
            )
        )
        
        # 3. Similar phrasing should hit
        hit = loop.run_until_complete(
            self.cache.get_cached_response("Explain quantum computing qubit superposition states")
        )
        self.assertIsNotNone(hit)
        self.assertTrue(hit["cache_hit"])
        self.assertGreaterEqual(hit["similarity"], 0.60)
        self.assertLess(hit["latency_ms"], 5.0)
        loop.close()

    def test_telemetry_stats(self):
        stats = self.cache.get_stats()
        self.assertIn("total_cached_prompts", stats)
        self.assertIn("total_cache_hits", stats)
        self.assertIn("estimated_tokens_saved", stats)

if __name__ == "__main__":
    unittest.main()
