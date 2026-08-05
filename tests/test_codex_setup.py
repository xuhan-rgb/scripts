import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
SETUP_SCRIPT = REPOSITORY / "claude" / "setup-codex.sh"
INSTALL_SCRIPT = REPOSITORY / "claude" / "install-codex-bridge.sh"
PROVIDER_SCRIPT = REPOSITORY / "claude" / "codex_provider.py"


class CodexSetupTests(unittest.TestCase):
    enabled_skills = (
        "agent-reach",
        "brainstorming",
        "grill-me",
        "grill-with-docs",
        "handoff",
        "tdd",
    )
    internal_skills = ("domain-modeling", "grilling")
    disabled_skills = ("dev-plan", "project-audit", "document-project", "unused")

    def test_claudex_disables_background_prompt_suggestions(self):
        installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--prompt-suggestions false", installer)
        self.assertIn("--model 'claudex-router[1m]'", installer)
        self.assertIn("--autocompact 250k", installer)
        self.assertIn("ANTHROPIC_CUSTOM_MODEL_OPTION=claudex-router[1m]", installer)
        self.assertIn("ANTHROPIC_CUSTOM_MODEL_OPTION_NAME=GPT Router (1M)", installer)
        self.assertIn("127.0.0.1:8320", installer)
        self.assertIn("availableModels", installer)
        self.assertNotIn("ANTHROPIC_DEFAULT_SONNET_MODEL", installer)
        self.assertNotIn("ANTHROPIC_DEFAULT_HAIKU_MODEL", installer)
        self.assertIn("extension_args=(--strict-mcp-config)", installer)
        self.assertNotIn("--safe-mode", installer)
        self.assertIn("CLAUDEX_EXTENSIONS", installer)
        self.assertIn('${CLAUDEX_YOLO:-0} == 1', installer)
        self.assertIn(
            '"skillOverrides":{"claude-api":"off"}',
            installer,
        )
        self.assertIn('--settings "${session_settings}"', installer)
        self.assertEqual(installer.count('skillOverrides\":{\"claude-api'), 1)
        sync = 'if sync_output="$(CLAUDEX_SYNC_BACKUP=1 "${BIN_DIR}/claude-codex-sync"'
        self.assertIn(sync, installer)
        self.assertLess(
            installer.index(sync),
            installer.index('systemctl --user restart "${SERVICE_NAME}"'),
        )
        self.assertIn('systemctl --user restart "${MANAGER_SERVICE_NAME}"', installer)
        self.assertIn('"${LIB_DIR}/codex_provider.py"', installer)
        self.assertIn('systemctl --user disable --now "${SERVICE_NAME}"', installer)
        self.assertNotIn('"${BIN_DIR}/codex-provider"', installer)

    def test_default_skill_allowlist_preserves_claude_memory_files(self):
        setup = SETUP_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'DEFAULT_ENABLED_SKILLS="agent-reach brainstorming grill-me grill-with-docs handoff tdd"',
            setup,
        )
        self.assertIn('ENABLED_SKILLS="${DEFAULT_ENABLED_SKILLS}"', setup)
        self.assertIn("CLAUDEX_AGENT_REACH", setup)
        self.assertIn(
            "Enable Agent Reach using a Python virtual environment? [y/N]",
            setup,
        )
        self.assertIn("import ensurepip, venv", setup)
        self.assertIn("Checking Claude plugin source:", setup)
        self.assertIn("Installing Claude plugin source:", setup)
        self.assertIn('INTERNAL_SKILLS="domain-modeling grilling"', setup)
        self.assertIn('readonly AGENT_REACH_VERSION="1.5.0"', setup)
        self.assertIn(
            "Panniantong/Agent-Reach/archive/refs/tags/v${AGENT_REACH_VERSION}.zip",
            setup,
        )
        self.assertNotIn('uv tool install "agent-reach==', setup)
        self.assertIn('alias claude-yolo=\'claude --dangerously-skip-permissions --strict-mcp-config\'', setup)
        self.assertIn('settings["skillOverrides"] = overrides', setup)
        self.assertIn('"name-only" if name in internal_skills', setup)
        self.assertNotIn("--safe-mode", setup)

    def test_fresh_setup_is_idempotent_and_keeps_claude_login_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bashrc = home / ".bashrc"
            legacy_provider = home / ".local" / "bin" / "codex-provider"
            legacy_provider.parent.mkdir(parents=True)
            legacy_provider.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            skill = home / ".agents" / "skills" / "example" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: example\n---\n", encoding="utf-8")
            for skill_name in self.enabled_skills + self.internal_skills + self.disabled_skills:
                skill_file = home / ".agents" / "skills" / skill_name / "SKILL.md"
                skill_file.parent.mkdir(parents=True, exist_ok=True)
                skill_file.write_text(
                    f"---\nname: {skill_name}\n---\n",
                    encoding="utf-8",
                )
            for skill_name in self.enabled_skills + self.internal_skills + self.disabled_skills:
                skill_file = home / ".claude" / "skills" / skill_name / "SKILL.md"
                skill_file.parent.mkdir(parents=True, exist_ok=True)
                skill_file.write_text(
                    f"---\nname: {skill_name}\n---\n",
                    encoding="utf-8",
                )
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
                    "CLAUDEX_AGENT_REACH": "0",
                    "CLAUDEX_SKIP_SKILL_INSTALL": "1",
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
                    config_path = home / ".codex" / "config.toml"
                    config_path.write_text(
                        config_path.read_text(encoding="utf-8")
                        .replace("model_context_window = 372000", "model_context_window = 999999")
                        .replace(
                            "model_auto_compact_token_limit = 244800",
                            "model_auto_compact_token_limit = 999999",
                        ),
                        encoding="utf-8",
                    )
                    with config_path.open("a", encoding="utf-8") as config_file:
                        config_file.write('\n[plugins."unused@example"]\nenabled = true\n')

            config = home / ".codex" / "config.toml"
            secrets = home / ".config" / "codex" / "secrets.env"
            config_text = config.read_text(encoding="utf-8")
            bashrc_text = bashrc.read_text(encoding="utf-8")

            self.assertIn('model_provider = "crs_local"', config_text)
            self.assertEqual(config_text.count("model_context_window = 372000"), 1)
            self.assertEqual(config_text.count("model_auto_compact_token_limit = 244800"), 1)
            self.assertIn("[model_providers.crs_local]", config_text)
            self.assertIn("[mcp_servers.openaiDeveloperDocs]", config_text)
            self.assertIn('[plugins."unused@example"]\nenabled = false', config_text)
            self.assertIn("enabled = false", config_text)
            self.assertIn(str(skill), config_text)
            self.assertIn("[[skills.config]]", config_text)
            for skill_name in self.enabled_skills + self.internal_skills:
                expected = "false" if skill_name == "agent-reach" else "true"
                self.assertIn(
                    'path = "'
                    + str(home / ".agents" / "skills" / skill_name / "SKILL.md")
                    + f'"\nenabled = {expected}',
                    config_text,
                )
            for skill_name in self.disabled_skills:
                self.assertIn(
                    'path = "'
                    + str(home / ".agents" / "skills" / skill_name / "SKILL.md")
                    + '"\nenabled = false',
                    config_text,
                )
            self.assertNotIn(secret_value, config_text)
            self.assertIn(secret_value, secrets.read_text(encoding="utf-8"))
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(secrets.stat().st_mode & 0o777, 0o600)
            settings = home / ".claude" / "settings.json"
            settings_data = json.loads(settings.read_text(encoding="utf-8"))
            for skill_name in self.enabled_skills:
                expected = "off" if skill_name == "agent-reach" else "on"
                self.assertEqual(settings_data["skillOverrides"][skill_name], expected)
            for skill_name in self.internal_skills:
                self.assertEqual(
                    settings_data["skillOverrides"][skill_name],
                    "name-only",
                )
            for skill_name in self.disabled_skills:
                self.assertEqual(settings_data["skillOverrides"][skill_name], "off")
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
            self.assertIn("alias claude-yolo='claude --dangerously-skip-permissions --strict-mcp-config'", bashrc_text)
            self.assertIn("alias claudex-yolo='CLAUDEX_YOLO=1 claudex'", bashrc_text)
            self.assertNotIn("claude-api", bashrc_text)
            self.assertNotIn("--safe-mode", bashrc_text)
            self.assertLess(
                output.index("[1/3] Initializing Codex providers"),
                output.index("[2/3] Updating aliases, skills, and extension policy"),
            )
            self.assertFalse(legacy_provider.exists())
            self.assertIn("Agent Reach disabled by CLAUDEX_AGENT_REACH=0", output)
            self.assertTrue((home / ".claude" / "settings.json").exists())

    def test_existing_active_provider_is_adopted_and_initialized(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            codex_home = home / ".codex"
            codex_home.mkdir()
            config = codex_home / "config.toml"
            config.write_text(
                'model_provider = "existing_route"\n'
                'model = "gpt-5.6-terra"\n'
                'model_reasoning_effort = "high"\n\n'
                '[model_providers.existing_route]\n'
                'name = "existing_route"\n'
                'base_url = "https://example.invalid/openai"\n'
                'wire_api = "responses"\n'
                'requires_openai_auth = false\n'
                'env_key = "EXISTING_ROUTE_KEY"\n',
                encoding="utf-8",
            )
            migration_secrets = home / "migration-secrets.env"
            migration_secrets.write_text(
                "export EXISTING_ROUTE_KEY='test-secret'\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.pop("EXISTING_ROUTE_KEY", None)
            environment.update(
                {
                    "HOME": str(home),
                    "CODEX_HOME": str(codex_home),
                    "CLAUDEX_NONINTERACTIVE": "1",
                    "CLAUDEX_AGENT_REACH": "0",
                    "CLAUDEX_SKIP_SKILL_INSTALL": "1",
                    "CLAUDEX_SECRETS_FILE": str(migration_secrets),
                }
            )

            completed = subprocess.run(
                ["bash", str(SETUP_SCRIPT)],
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Codex configured: provider=existing_route", completed.stdout)
            self.assertIn('model_provider = "existing_route"', config.read_text(encoding="utf-8"))
            profile = codex_home / "existing_route.config.toml"
            self.assertTrue(profile.is_file())
            profile_text = profile.read_text(encoding="utf-8")
            self.assertIn('model_provider = "existing_route"', profile_text)
            self.assertIn('model = "gpt-5.6-terra"', profile_text)
            self.assertIn('model_reasoning_effort = "high"', profile_text)

    def test_setup_defers_a_missing_provider_key_to_the_web_console(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            environment = os.environ.copy()
            environment.pop("CRS_OPENAI_KEY", None)
            environment.update(
                {
                    "HOME": str(home),
                    "CODEX_HOME": str(home / ".codex"),
                    "CLAUDEX_NONINTERACTIVE": "1",
                    "CLAUDEX_AGENT_REACH": "0",
                    "CLAUDEX_SKIP_SKILL_INSTALL": "1",
                }
            )

            completed = subprocess.run(
                ["bash", str(SETUP_SCRIPT)],
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("has no Key yet; configure it at http://127.0.0.1:8320", completed.stdout)
            self.assertTrue((home / ".bashrc").is_file())
            self.assertTrue((home / ".codex" / "config.toml").is_file())

    def test_codex_provider_accepts_a_key_over_stdin_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            codex_home = home / ".codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                '[model_providers.test_route]\n'
                'base_url = "https://gateway.example/openai"\n'
                'wire_api = "responses"\n'
                'env_key = "TEST_ROUTE_KEY"\n',
                encoding="utf-8",
            )
            secret = "secret-never-echo"
            environment = os.environ.copy()
            environment.update({"HOME": str(home), "CODEX_HOME": str(codex_home)})

            completed = subprocess.run(
                ["python3", str(PROVIDER_SCRIPT), "set-key", "test_route", "--stdin"],
                input=secret,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn(secret, completed.stdout + completed.stderr)
            secrets = home / ".config" / "codex" / "secrets.env"
            self.assertIn(secret, secrets.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
