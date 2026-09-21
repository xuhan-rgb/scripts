"""Independent preview checks; no access to real account data."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt5.QtWidgets import QApplication

spec = importlib.util.spec_from_file_location("preview_table", Path(__file__).resolve().parents[1] / "preview-token-table.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class TimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_real_format_requests_results_and_pending_call(self):
        def event(kind, **payload):
            return {"type": kind, "payload": payload}
        usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20, "total_tokens": 120}
        record = event("token_usage_record", turn_id="new", response_id="r1", usage=usage)
        events = [event("turn_context", turn_id="old", model="old-model"),
                  event("token_usage_record", turn_id="old", response_id="old", usage=usage),
                  event("response_item", role="user", content=[{"type": "input_text", "text": "真实格式问题"}],
                        internal_chat_message_metadata_passthrough={"turn_id": "new"}),
                  event("turn_context", turn_id="new", model="model-a"),
                  event("response_item", type="custom_tool_call", name="exec", call_id="c1",
                        input='text(await tools.exec_command({cmd:"echo hello"}));'), record, record,
                  event("response_item", type="custom_tool_call_output", call_id="c1", output="hello"),
                  event("response_item", type="custom_tool_call", name="exec", call_id="c2",
                        input='text(await tools.apply_patch("patch"));')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in events) + '{"partial":')
            report = preview.read_report(path)
        self.assertEqual(report["question"], "真实格式问题")
        self.assertEqual(report["tool_count"], 2)
        self.assertEqual(len(report["steps"]), 2)
        self.assertEqual(report["steps"][0]["usage"]["total_tokens"], 120)
        self.assertEqual(report["steps"][0]["title"], "执行本地命令")
        self.assertIn("已收到 1 次返回", report["steps"][0]["result"])
        self.assertIn("hello", report["steps"][0]["details"])
        self.assertIsNone(report["steps"][1]["usage"])
        window = preview.Preview()
        window.timer.stop()
        window.render(report)
        self.assertTrue(window.details.isHidden())
        self.assertEqual(window.table.item(0, 1).text(), "执行本地命令")
        self.assertEqual(window.table.item(0, 5).text(), "120")
        self.assertIn("80.0%", window.summary.text())
        self.assertIn("等待", window.table.item(1, 2).text())
        window.table.cellWidget(0, 7).click()
        updated = {**report, "steps": [dict(step) for step in report["steps"]]}
        updated["steps"][1]["usage"] = usage
        window.render(updated)
        self.assertFalse(window.details.isHidden())
        self.assertIn("hello", window.details.toPlainText())
        self.assertEqual(window.table.item(1, 5).text(), "120")
        self.assertIn("240 token", window.summary.text())
        window.table.cellWidget(0, 7).click()
        self.assertTrue(window.details.isHidden())
        window.close()


if __name__ == "__main__":
    unittest.main()
