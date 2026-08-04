import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "claude" / "codex_bridge_manager.py"
SPEC = importlib.util.spec_from_file_location("codex_bridge_manager", MODULE_PATH)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


SAMPLE_CONFIG = '''model_provider = "provider_b"
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"

[model_providers.provider_a]
base_url = "https://a.example/openai"
wire_api = "responses"
env_key = "A_KEY"

[model_providers.provider_b]
base_url = "http://127.0.0.1:3000/openai"
wire_api = "responses"
env_key = "B_KEY"
'''


class CodexBridgeManagerTests(unittest.TestCase):
    def test_reads_the_dynamically_selected_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text(SAMPLE_CONFIG, encoding="utf-8")

            state = manager.read_codex_state(config)

        self.assertEqual(state["provider"], "provider_b")
        self.assertEqual(state["base_url"], "http://127.0.0.1:3000/openai")
        self.assertEqual(state["env_key"], "B_KEY")

    def test_selection_is_independent_from_codex_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.conf"
            lock = Path(directory) / "selection.lock"
            manager.write_selection("gpt-5.6-terra", "high", selection, lock)

            result = manager.read_selection(
                selection,
                {"codex_model": "gpt-5.6-sol", "codex_effort": "xhigh"},
            )

        self.assertEqual(result, {"model": "gpt-5.6-terra", "effort": "high"})

    def test_rejects_unknown_model_and_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.conf"
            lock = Path(directory) / "selection.lock"
            with self.assertRaisesRegex(ValueError, "unsupported model"):
                manager.write_selection("gpt-image-2", "high", selection, lock)
            with self.assertRaisesRegex(ValueError, "unsupported effort"):
                manager.write_selection("gpt-5.6-sol", "ultra", selection, lock)


if __name__ == "__main__":
    unittest.main()
