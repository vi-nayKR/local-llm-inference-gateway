import unittest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gateway.vllm_client import LocalInferenceClient

class TestLocalInferenceProxy(unittest.TestCase):
    def setUp(self):
        self.client = LocalInferenceClient(default_model="meta-llama/Llama-3.2-1B-Instruct")

    def test_local_slm_generation(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        messages = [{"role": "user", "content": "Explain how PagedAttention operates"}]
        result = loop.run_until_complete(
            self.client.generate_completion(messages=messages)
        )
        
        self.assertIn("content", result)
        self.assertIn("vLLM's PagedAttention", result["content"])
        self.assertEqual(result["model"], "meta-llama/Llama-3.2-1B-Instruct")
        self.assertGreater(result["tokens_per_second"], 50.0)
        self.assertEqual(result["finish_reason"], "stop")
        loop.close()

    def test_sse_token_streaming(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        messages = [{"role": "user", "content": "What is QLoRA fine-tuning?"}]
        chunks = []
        
        async def collect_stream():
            async for chunk in self.client.stream_completion(messages=messages):
                chunks.append(chunk)

        loop.run_until_complete(collect_stream())
        
        self.assertGreater(len(chunks), 5)
        self.assertTrue(any("data: {" in c for c in chunks))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        loop.close()

if __name__ == "__main__":
    unittest.main()
