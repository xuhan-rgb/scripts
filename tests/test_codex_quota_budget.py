import importlib.util
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
BACKEND_PATH = REPOSITORY / "claude" / "codex_account_manager_backend.py"
SPEC = importlib.util.spec_from_file_location("codex_account_manager_backend", BACKEND_PATH)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


class CodexQuotaBudgetTests(unittest.TestCase):
    NOW = 1_000_000.0

    def window(self, remaining_percent, seconds_left, window_seconds=604800):
        return {
            "window_seconds": window_seconds,
            "resets_at": self.NOW + seconds_left,
            "remaining_percent": remaining_percent,
        }

    def test_seven_day_window_reports_daily_budget_and_normal_status(self):
        result = backend.calculate_weekly_budget(
            self.window(98, 166 * 3600), now=self.NOW
        )

        self.assertEqual(result["remaining_percent"], 98.0)
        self.assertAlmostEqual(result["target_percent"], 98.8095238095)
        self.assertAlmostEqual(result["difference_percent"], -0.8095238095)
        self.assertAlmostEqual(result["budget_percent"], 14.1686746988)
        self.assertEqual(result["status"], "normal")
        self.assertEqual(result["label"], "Daily budget")

    def test_consuming_faster_than_target_is_warned(self):
        result = backend.calculate_weekly_budget(
            self.window(30, 96 * 3600), now=self.NOW
        )

        self.assertAlmostEqual(result["target_percent"], 57.1428571429)
        self.assertAlmostEqual(result["difference_percent"], -27.1428571429)
        self.assertAlmostEqual(result["budget_percent"], 7.5)
        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["label"], "Daily budget")

    def test_pause_reaches_exact_baseline_without_usage(self):
        window = self.window(75, 144 * 3600)
        budget = backend.calculate_weekly_budget(window, now=self.NOW)
        self.assertEqual(budget["pause_seconds"], 18 * 3600)
        caught_up = backend.calculate_weekly_budget(window, now=self.NOW + budget["pause_seconds"])
        self.assertEqual(caught_up["difference_percent"], 0)
        self.assertEqual(caught_up["pause_seconds"], 0)
        self.assertEqual(backend.format_pace_pause(caught_up), "")
        lower = backend.calculate_weekly_budget(self.window(74, 144 * 3600), now=self.NOW)
        self.assertGreater(lower["pause_seconds"], budget["pause_seconds"])

    def test_pause_rounds_up_and_handles_expired_or_ahead(self):
        budget = backend.calculate_weekly_budget(self.window(50, 84 * 3600 + 1), now=self.NOW)
        self.assertEqual(backend.format_pace_pause(budget), "Pause 0h 1m to get on pace")
        for remaining, seconds in ((90, 86400), (0, 0), (50, -60)):
            budget = backend.calculate_weekly_budget(self.window(remaining, seconds), now=self.NOW)
            self.assertEqual(budget["pause_seconds"], 0)
            self.assertEqual(backend.format_pace_pause(budget), "")

    def test_zero_balance_is_empty(self):
        result = backend.calculate_weekly_budget(
            self.window(0, 96 * 3600), now=self.NOW
        )

        self.assertEqual(result["remaining_percent"], 0.0)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["label"], "Daily budget")

    def test_less_than_one_day_uses_until_reset_label_and_one_day_divisor(self):
        result = backend.calculate_weekly_budget(
            self.window(20, 12 * 3600), now=self.NOW
        )

        self.assertAlmostEqual(result["budget_percent"], 20.0)
        self.assertEqual(result["label"], "Until reset")
        self.assertEqual(result["status"], "normal")

    def test_expired_window_has_no_budget_and_refresh_status(self):
        result = backend.calculate_weekly_budget(
            self.window(20, 0), now=self.NOW
        )

        self.assertEqual(result["target_percent"], 0.0)
        self.assertEqual(result["difference_percent"], 20.0)
        self.assertIsNone(result["budget_percent"])
        self.assertEqual(result["status"], "refresh")
        self.assertEqual(result["label"], "Until reset")

    def test_remaining_time_is_clamped_to_week(self):
        result = backend.calculate_weekly_budget(
            self.window(50, 10 * 86400), now=self.NOW
        )

        self.assertEqual(result["target_percent"], 100.0)
        self.assertAlmostEqual(result["difference_percent"], -50.0)

    def test_difference_of_exactly_one_percent_is_normal(self):
        result = backend.calculate_weekly_budget(
            self.window(50, 84 * 3600), now=self.NOW
        )

        self.assertEqual(result["target_percent"], 50.0)
        self.assertEqual(result["difference_percent"], 0.0)
        self.assertEqual(result["status"], "normal")

        result = backend.calculate_weekly_budget(
            self.window(49, 84 * 3600), now=self.NOW
        )
        self.assertEqual(result["difference_percent"], -1.0)
        self.assertEqual(result["status"], "normal")

    def test_difference_below_one_percent_is_warned(self):
        result = backend.calculate_weekly_budget(
            self.window(48.999, 84 * 3600), now=self.NOW
        )

        self.assertLess(result["difference_percent"], -1.0)
        self.assertEqual(result["status"], "warn")

    def test_non_weekly_windows_return_none(self):
        self.assertIsNone(
            backend.calculate_weekly_budget(self.window(50, 86400, 86400), now=self.NOW)
        )

    def test_invalid_or_non_finite_values_return_none(self):
        invalid_windows = [
            self.window(-1, 86400),
            self.window(101, 86400),
            self.window(float("nan"), 86400),
            self.window(float("inf"), 86400),
            self.window(50, 86400),
            self.window(50, 86400),
        ]
        invalid_windows[-2]["resets_at"] = float("nan")
        invalid_windows[-1]["resets_at"] = float("inf")

        for window in invalid_windows:
            with self.subTest(window=window):
                self.assertIsNone(
                    backend.calculate_weekly_budget(window, now=self.NOW)
                )

        for remaining in (True, False, "50", None):
            with self.subTest(remaining=remaining):
                self.assertIsNone(
                    backend.calculate_weekly_budget(
                        self.window(remaining, 86400), now=self.NOW
                    )
                )

        for reset_at in (True, False, "1000001", None):
            window = self.window(50, 86400)
            window["resets_at"] = reset_at
            with self.subTest(reset_at=reset_at):
                self.assertIsNone(
                    backend.calculate_weekly_budget(window, now=self.NOW)
                )

if __name__ == "__main__":
    unittest.main()
