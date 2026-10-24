"""Free checks: actual SDK serialization with mock HTTP and real RAPTOR logic.

Run from the repository root:
    .venv/bin/python -m unittest discover -s tests -v
Only the cl100k_base tokenizer data must already be cached. Network access is
blocked before importing RAPTOR; no model weights or API keys are needed.
"""

import hashlib
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "local/tiktoken-cache"))
os.environ.setdefault("HF_HOME", str(ROOT / "local/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def deny_network(*args, **kwargs):
    raise AssertionError("Network is forbidden in offline tests")


# Block sockets even during imports (including an accidental cache download).
NETWORK_PATCHES = [patch("socket.socket.connect", deny_network),
                   patch("socket.create_connection", deny_network),
                   patch("socket.getaddrinfo", deny_network)]
for network_patch in NETWORK_PATCHES:
    network_patch.start()

import httpx
from openai import AuthenticationError, OpenAI

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels_cus import BailianEmbeddingModel
from raptor.QAModels_cus import DeepSeekQAModel
from raptor.SummarizationModels_cus import DeepSeekSummarizationModel

logging.disable(logging.INFO)


def tearDownModule():
    for network_patch in reversed(NETWORK_PATCHES):
        network_patch.stop()


class CustomModelTests(unittest.TestCase):
    def setUp(self):
        # Tests must be independent of the user's shell and .env model choices.
        model_env = {name: "" for name in (
            "DEEPSEEK_MODEL", "DEEPSEEK_SUMMARIZATION_MODEL", "DEEPSEEK_QA_MODEL",
            "DASHSCOPE_EMBEDDING_MODEL", "DASHSCOPE_EMBEDDING_DIMENSIONS",
            "DEEPSEEK_QA_MAX_TOKENS",
        )}
        self.environment = patch.dict(os.environ, model_env)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.requests = []
        self.content = " Cinderella married the prince. "
        self.vector = None
        self.status = 200
        self.fail_once = False
        self.client = OpenAI(
            api_key="offline-placeholder",
            base_url="https://offline.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            max_retries=0,
        )
        self.addCleanup(self.client.close)

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if self.status != 200:
            status = self.status
            if self.fail_once:
                self.status = 200
            return httpx.Response(status, json={"error": {"message": "mock error"}},
                                  headers={"retry-after": "0.001"})
        if request.url.path.endswith("/embeddings"):
            vector = self.vector
            if vector is None:
                digest = hashlib.sha256(body["input"][0].encode()).digest()
                vector = [(digest[i % len(digest)] + i % 7 + 1) / 263
                          for i in range(body.get("dimensions", 1536))]
            return httpx.Response(200, json={
                "object": "list", "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": self.content,
            }}],
        })

    def models(self):
        return (
            DeepSeekSummarizationModel(client=self.client),
            DeepSeekQAModel(client=self.client, max_tokens=80),
            BailianEmbeddingModel(client=self.client, dimensions=64),
        )

    def test_summary_http_contract(self):
        summary, _, _ = self.models()
        self.assertEqual(summary.summarize("A story", 100), self.content.strip())
        path, body = self.requests[-1]
        self.assertTrue(path.endswith("/chat/completions"))
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 100)
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_qa_uses_context_question_and_output_limit(self):
        _, qa, _ = self.models()
        self.assertEqual(qa.answer_question("The story", "Who married?"), self.content.strip())
        body = self.requests[-1][1]
        self.assertIn("The story", body["messages"][-1]["content"])
        self.assertIn("Who married?", body["messages"][-1]["content"])
        self.assertEqual(body["max_tokens"], 80)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_embedding_http_contract_and_float_output(self):
        _, _, embedding = self.models()
        vector = embedding.create_embedding("A\nstory")
        path, body = self.requests[-1]
        self.assertTrue(path.endswith("/embeddings"))
        self.assertEqual(body, {"input": ["A story"], "model": "text-embedding-v4",
                                "encoding_format": "float", "dimensions": 64})
        self.assertEqual(len(vector), 64)
        self.assertTrue(all(type(v) is float for v in vector))

    def test_blank_chat_response_fails(self):
        summary, qa, _ = self.models()
        self.content = " "
        with self.assertRaisesRegex(ValueError, "empty summary"):
            summary.summarize("story")
        with self.assertRaisesRegex(ValueError, "empty answer"):
            qa.answer_question("story", "question")

    def test_bad_vectors_fail(self):
        _, _, embedding = self.models()
        for vector in ([1.0], [0.0] * 64, [None] * 64):
            with self.subTest(vector_kind=type(vector[0]).__name__):
                self.vector = vector
                with self.assertRaises(ValueError):
                    embedding.create_embedding("story")

    def test_input_validation_makes_no_requests(self):
        summary, qa, embedding = self.models()
        for call in (lambda: summary.summarize(" "),
                     lambda: summary.summarize("story", 0),
                     lambda: summary.summarize("story", True),
                     lambda: qa.answer_question("story", ""),
                     lambda: qa.answer_question("", "question"),
                     lambda: embedding.create_embedding(""),
                     lambda: BailianEmbeddingModel(client=self.client, dimensions=65),
                     lambda: BailianEmbeddingModel(client=self.client, max_concurrency=0)):
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(self.requests, [])

    def test_missing_configuration_fails_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            for constructor in (DeepSeekSummarizationModel, DeepSeekQAModel,
                                BailianEmbeddingModel):
                with self.assertRaisesRegex(ValueError, "API_KEY"):
                    constructor()
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-placeholder"}):
                with self.assertRaisesRegex(ValueError, "DASHSCOPE_BASE_URL"):
                    BailianEmbeddingModel()

    def test_real_client_initialization_and_environment_configuration(self):
        env = {"DEEPSEEK_API_KEY": "offline-placeholder",
               "DASHSCOPE_API_KEY": "offline-placeholder",
               "DEEPSEEK_MODEL": "custom-deepseek-model",
               "DASHSCOPE_BASE_URL": "https://offline.invalid/compatible-mode/v1"}
        with patch.dict(os.environ, env):
            for constructor in (DeepSeekSummarizationModel, DeepSeekQAModel,
                                BailianEmbeddingModel):
                model = constructor()
                self.addCleanup(model.client.close)
                self.assertEqual(model.client.max_retries, 2)
                if not isinstance(model, BailianEmbeddingModel):
                    self.assertEqual(model.model, "custom-deepseek-model")

    def test_per_task_models_and_parameters_are_sent_from_environment(self):
        env = {"DEEPSEEK_MODEL": "shared-fallback",
               "DEEPSEEK_SUMMARIZATION_MODEL": "summary-model",
               "DEEPSEEK_QA_MODEL": "qa-model",
               "DEEPSEEK_QA_MAX_TOKENS": "120",
               "DASHSCOPE_EMBEDDING_MODEL": "text-embedding-v3",
               "DASHSCOPE_EMBEDDING_DIMENSIONS": "256"}
        with patch.dict(os.environ, env):
            summary = DeepSeekSummarizationModel(client=self.client)
            qa = DeepSeekQAModel(client=self.client)
            embedding = BailianEmbeddingModel(client=self.client)
            summary.summarize("story")
            self.assertEqual(self.requests[-1][1]["model"], "summary-model")
            qa.answer_question("story", "question")
            self.assertEqual(self.requests[-1][1]["model"], "qa-model")
            self.assertEqual(self.requests[-1][1]["max_tokens"], 120)
            self.assertEqual(len(embedding.create_embedding("story")), 256)
            self.assertEqual(self.requests[-1][1]["model"], "text-embedding-v3")
            self.assertEqual(self.requests[-1][1]["dimensions"], 256)
            self.assertEqual(DeepSeekQAModel("explicit-model", client=self.client,
                                            max_tokens=40).max_tokens, 40)
            self.assertEqual(DeepSeekQAModel("explicit-model", client=self.client).model,
                             "explicit-model")

    def test_invalid_environment_parameters_fail_before_api_call(self):
        for env, constructor in (
            ({"DASHSCOPE_EMBEDDING_DIMENSIONS": "invalid"}, BailianEmbeddingModel),
            ({"DASHSCOPE_EMBEDDING_DIMENSIONS": "0"}, BailianEmbeddingModel),
            ({"DEEPSEEK_QA_MAX_TOKENS": "invalid"}, DeepSeekQAModel),
            ({"DEEPSEEK_QA_MAX_TOKENS": "0"}, DeepSeekQAModel),
        ):
            with self.subTest(variable=next(iter(env))), patch.dict(os.environ, env):
                with self.assertRaises(ValueError):
                    constructor(client=self.client)
        self.assertEqual(self.requests, [])

    def test_fixed_dimension_embedding_models_omit_dimensions_parameter(self):
        embedding = BailianEmbeddingModel("text-embedding-v2", client=self.client)
        self.assertEqual(len(embedding.create_embedding("story")), 1536)
        self.assertNotIn("dimensions", self.requests[-1][1])

    def test_authentication_error_is_not_retried(self):
        self.status = 401
        self.client.max_retries = 2
        with self.assertRaises(AuthenticationError):
            self.models()[0].summarize("story")
        self.assertEqual(len(self.requests), 1)

    def test_rate_limit_is_retried_with_a_bound(self):
        self.status = 429
        self.fail_once = True
        self.client.max_retries = 1
        self.models()[0].summarize("story")
        self.assertEqual(len(self.requests), 2)

    def test_cinderella_build_retrieve_answer_save_reload(self):
        summary, qa, embedding = self.models()
        # Default OpenAI or local model construction must never occur.
        with patch("raptor.EmbeddingModels.OpenAI", side_effect=AssertionError("default model")), \
             patch("raptor.QAModels.OpenAI", side_effect=AssertionError("default model")), \
             patch("raptor.EmbeddingModels.SentenceTransformer", side_effect=AssertionError("local model")), \
             patch("raptor.QAModels.T5ForConditionalGeneration.from_pretrained", side_effect=AssertionError("local model")):
            config = RetrievalAugmentationConfig(
                summarization_model=summary, qa_model=qa, embedding_model=embedding)
            self.assertIs(config.tree_retriever_config.embedding_model, embedding)
            self.assertEqual(config.tree_builder_config.embedding_models, {"EMB": embedding})
            self.assertEqual(config.tree_builder_config.max_tokens, 100)
            self.assertEqual(config.tree_builder_config.num_layers, 5)
            self.assertEqual(config.tree_builder_config.reduction_dimension, 10)
            ra = RetrievalAugmentation(config=config)
            ra.add_documents((ROOT / "demo/sample.txt").read_text(encoding="utf-8"))
            self.assertGreater(ra.tree.num_layers, 0)
            self.assertGreater(len(ra.tree.all_nodes), len(ra.tree.leaf_nodes))
            question = "How did Cinderella reach her happy ending?"
            context, layers = ra.retrieve(question)
            self.assertTrue(context.strip())
            self.assertTrue(layers)
            self.assertEqual(ra.answer_question(question), self.content.strip())
            for node in ra.tree.all_nodes.values():
                self.assertEqual(len(node.embeddings["EMB"]), 64)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "tree.pkl")
                ra.save(path)
                reloaded = RetrievalAugmentation(config=config, tree=path)
                self.assertEqual(reloaded.answer_question(question), self.content.strip())

    def test_notebook_api_cells_and_index_reuse_with_mock_http(self):
        notebook = json.loads((ROOT / "demo_cus.ipynb").read_text())
        env = {"DEEPSEEK_API_KEY": "offline-placeholder",
               "DASHSCOPE_API_KEY": "offline-placeholder",
               "DASHSCOPE_BASE_URL": "https://offline.invalid/v1"}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, env), \
             contextlib.redirect_stdout(io.StringIO()), \
             patch("raptor.SummarizationModels_cus.DeepSeekSummarizationModel",
                   side_effect=lambda **kw: DeepSeekSummarizationModel(client=self.client, **kw)), \
             patch("raptor.QAModels_cus.DeepSeekQAModel",
                   side_effect=lambda **kw: DeepSeekQAModel(client=self.client, **kw)), \
             patch("raptor.EmbeddingModels_cus.BailianEmbeddingModel",
                   side_effect=lambda **kw: BailianEmbeddingModel(client=self.client, **kw)):
            for run in range(2):
                if run:
                    # Changing only QA must not invalidate the cached tree.
                    os.environ["DEEPSEEK_QA_MODEL"] = "another-qa-model"
                previous_requests = len(self.requests)
                namespace = {"__name__": "__main__"}
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    source = source.replace("RUN_PAID_API = False", "RUN_PAID_API = True")
                    if run:
                        source = source.replace("RUN_API_SMOKE_TEST = True", "RUN_API_SMOKE_TEST = False")
                    exec(compile(source, f"demo_cus.ipynb:cell{index}", "exec"), namespace)
                    if "REBUILD_TREE = False" in source:
                        namespace["SAVE_PATH"] = Path(directory) / "tree.pkl"
                        namespace["META_PATH"] = Path(directory) / "tree.json"
                self.assertEqual(namespace["answer"], self.content.strip())
                self.assertEqual(namespace["embedding"].dimensions, 1024)
                self.assertGreater(namespace["RA"].tree.num_layers, 0)
                self.assertEqual(len(namespace["RA_loaded"].tree.all_nodes),
                                 len(namespace["RA"].tree.all_nodes))
                if run:
                    # Cached tree: one query embedding and one QA, no rebuilding.
                    self.assertEqual(len(self.requests) - previous_requests, 2)


if __name__ == "__main__":
    unittest.main()
