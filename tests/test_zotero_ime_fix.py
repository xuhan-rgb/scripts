import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
FIX_SCRIPT = REPOSITORY / "desktop" / "fix-zotero-ime-candidate-position.sh"


class ZoteroImeFixTests(unittest.TestCase):
    def test_repairs_all_profiles_without_touching_other_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_root = Path(directory) / "zotero"
            first_profile = profile_root / "first.default"
            second_profile = profile_root / "second.default"
            first_profile.mkdir(parents=True)
            second_profile.mkdir(parents=True)

            prefs = first_profile / "prefs.js"
            prefs.write_text(
                'user_pref("another.preference", true);\n'
                'user_pref("focusmanager.testmode", true);\n',
                encoding="utf-8",
            )
            user = first_profile / "user.js"
            user.write_text(
                '  user_pref( "focusmanager.testmode" , true );  \n',
                encoding="utf-8",
            )
            already_fixed = second_profile / "prefs.js"
            already_fixed.write_text(
                'user_pref("focusmanager.testmode", false);\n',
                encoding="utf-8",
            )

            command = [
                str(FIX_SCRIPT),
                "--profile-root",
                str(profile_root),
                "--no-process-control",
            ]
            first_run = subprocess.run(command, capture_output=True, text=True)

            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertIn("已修复 2 个配置文件", first_run.stdout)
            self.assertEqual(
                prefs.read_text(encoding="utf-8"),
                'user_pref("another.preference", true);\n'
                'user_pref("focusmanager.testmode", false);\n',
            )
            self.assertEqual(
                user.read_text(encoding="utf-8"),
                '  user_pref( "focusmanager.testmode" , false );  \n',
            )
            self.assertEqual(
                already_fixed.read_text(encoding="utf-8"),
                'user_pref("focusmanager.testmode", false);\n',
            )
            backups_after_first_run = sorted(profile_root.rglob("*.bak.*"))
            self.assertEqual(len(backups_after_first_run), 2)

            second_run = subprocess.run(command, capture_output=True, text=True)

            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertIn("无需修改", second_run.stdout)
            self.assertEqual(
                sorted(profile_root.rglob("*.bak.*")),
                backups_after_first_run,
            )

    def test_missing_profile_root_fails_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_root = Path(directory) / "missing"

            completed = subprocess.run(
                [
                    str(FIX_SCRIPT),
                    "--profile-root",
                    str(missing_root),
                    "--no-process-control",
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("找不到 Zotero profile 目录", completed.stderr)
            self.assertFalse(missing_root.exists())


if __name__ == "__main__":
    unittest.main()
