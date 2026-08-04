import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
SETUP_SCRIPT = REPOSITORY / "claude" / "setup-codex.sh"
INSTALL_SCRIPT = REPOSITORY / "claude" / "install-codex-bridge.sh"


class CodexSetupTests(unittest.TestCase):
    def test_claudex_disables_background_prompt_suggestions(self):
        installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--prompt-suggestions false", installer)
        self.assertIn("extension_args=(--safe-mode)", installer)
        self.assertNotIn("--disable-slash-commands", installer)
        self.assertIn("CLAUDEX_EXTENSIONS", installer)

    def test_fresh_setup_is_idempotent_and_keeps_claude_login_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bashrc = home / ".bashrc"
            skill = home / ".agents" / "skills" / "example" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: example\n---\n", encoding="utf-8")
            bashrc.write_text(
                "# keep this line\n"
                "alias codex-yolo='old-codex-command'\n"
                "  alias claude-yolo=\"old-claude-command\"\n"
                "alias claudex-yolo='old-claudex-command'\n",
                encoding="utf-8",
            )
            migration_secrets = home / "migration-secrets.env"
            secret_value = "test-secret-never-store-in-config"
            migration_secrets.write_text(
                f"export CRS_OPENAI_KEY='{secret_value}'\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            for name in ("CRS_OPENAI_KEY", "SORRYIOS_OPENAI_KEY", "ZSKJ_OPENAI_KEY"):
                environment.pop(name, None)
            environment.update(
                {
                    "HOME": str(home),
                    "CODEX_HOME": str(home / ".codex"),
                    "CLAUDEX_NONINTERACTIVE": "1",
                    "CLAUDEX_SECRETS_FILE": str(migration_secrets),
                }
            )

            output = ""
            for iteration in range(2):
                completed = subprocess.run(
                    ["bash", str(SETUP_SCRIPT)],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                output = completed.stdout
                if iteration == 0:
                    with (home / ".codex" / "config.toml").open("a", encoding="utf-8") as config_file:
                        config_file.write('\n[plugins."unused@example"]\nenabled = true\n')

            config = home / ".codex" / "config.toml"
            secrets = home / ".config" / "codex" / "secrets.env"
            config_text = config.read_text(encoding="utf-8")
            bashrc_text = bashrc.read_text(encoding="utf-8")

            self.assertIn('model_provider = "crs_local"', config_text)
            self.assertIn("[model_providers.crs_local]", config_text)
            self.assertIn("[mcp_servers.openaiDeveloperDocs]", config_text)
            self.assertIn('[plugins."unused@example"]\nenabled = false', config_text)
            self.assertIn("enabled = false", config_text)
            self.assertIn(str(skill), config_text)
            self.assertIn("[[skills.config]]", config_text)
            self.assertNotIn(secret_value, config_text)
            self.assertIn(secret_value, secrets.read_text(encoding="utf-8"))
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(secrets.stat().st_mode & 0o777, 0o600)
            self.assertEqual(bashrc_text.count("alias codex-yolo="), 1)
            self.assertEqual(bashrc_text.count("alias claude-yolo="), 1)
            self.assertEqual(bashrc_text.count("alias claudex-yolo="), 1)
            self.assertIn("# keep this line", bashrc_text)
            self.assertNotIn("old-codex-command", bashrc_text)
            self.assertNotIn("old-claude-command", bashrc_text)
            self.assertNotIn("old-claudex-command", bashrc_text)
            self.assertEqual(
                bashrc_text.count(
                    '[ -f "$HOME/.config/codex/secrets.env" ] && source "$HOME/.config/codex/secrets.env"'
                ),
                1,
            )
            self.assertIn("alias claude-yolo='claude --dangerously-skip-permissions --safe-mode'", bashrc_text)
            self.assertNotIn("--disable-slash-commands", bashrc_text)
            self.assertLess(output.index("[1/3]"), output.index("[2/3]"))
            self.assertFalse((home / ".claude").exists())


if __name__ == "__main__":
    unittest.main()
