import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).parents[1]
SETUP_SCRIPT = REPOSITORY / "claude" / "setup-codex.sh"
INSTALL_SCRIPT = REPOSITORY / "claude" / "install-codex-bridge.sh"
AUTH_SWITCH_SCRIPT = REPOSITORY / "claude" / "switch-codex-auth.sh"
PROVIDER_SCRIPT = REPOSITORY / "claude" / "codex_provider.py"
PROVIDER_SPEC = importlib.util.spec_from_file_location("codex_provider", PROVIDER_SCRIPT)
provider = importlib.util.module_from_spec(PROVIDER_SPEC)
PROVIDER_SPEC.loader.exec_module(provider)


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, content_type="application/json"):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Type": content_type}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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

    def test_provider_live_test_uses_the_selected_model_and_reads_streamed_answer(self):
        stream = (
            b'data: {"type":"response.output_text.delta","delta":"O"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"K"}\n\n'
            b'data: {"type":"response.completed"}\n\n'
        )
        response = FakeResponse(stream, content_type="text/event-stream")
        with (
            mock.patch.object(provider.socket, "create_connection", return_value=nullcontext()),
            mock.patch.object(provider, "urlopen", return_value=response) as open_url,
        ):
            ok, messages = provider.test_provider_connection(
                "http://127.0.0.1:3000/openai",
                "secret-never-logged",
                model="gpt-5.6-sol",
            )

        self.assertTrue(ok)
        self.assertIn("model: gpt-5.6-sol", messages)
        self.assertIn("answer: OK", messages)
        open_url.assert_called_once()
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertTrue(payload["stream"])
        self.assertNotIn("secret-never-logged", request.data.decode())
        self.assertNotIn("secret-never-logged", json.dumps(messages))

    def test_claudex_disables_background_prompt_suggestions(self):
        installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--prompt-suggestions false", installer)
        self.assertIn("--model 'claudex-router[1m]'", installer)
        self.assertIn("compact_args+=(--autocompact 250k)", installer)
        self.assertIn("grep -q -- '--autocompact'", installer)
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
        self.assertIn("settings.claudex-yolo.json", installer)
        self.assertIn("settings_args=(", installer)
        self.assertIn("settings_args[@]", installer)
        self.assertIn('--settings "${session_settings}"', installer)
        self.assertNotIn('skillOverrides\":{\"claude-api', installer)
        sync = 'if sync_output="$(CLAUDEX_SYNC_BACKUP=1 "${BIN_DIR}/claude-codex-sync"'
        self.assertIn(sync, installer)
        self.assertLess(
            installer.index(sync),
            installer.index('systemctl --user restart "${SERVICE_NAME}"'),
        )
        manager_install = 'install -m 0755 "${SCRIPT_DIR}/codex_bridge_manager.py"'
        manager_restart = 'systemctl --user restart "${MANAGER_SERVICE_NAME}"'
        self.assertIn(manager_restart, installer)
        self.assertLess(installer.index(manager_install), installer.index(manager_restart))
        self.assertIn('Restarting Codex Routing Desk with the installed web console', installer)
        self.assertIn('id="provider-config-open"', installer)
        self.assertIn('restarted with Provider config enabled', installer)
        self.assertIn('Restarting Claude-to-Codex gateway on 127.0.0.1:8317', installer)
        self.assertIn('Claude-to-Codex gateway restarted and ready on 127.0.0.1:8317', installer)
        self.assertIn('cmp -s "${tmp_dir}/claudex" "${BIN_DIR}/claudex"', installer)
        self.assertIn('cmp -s "${tmp_dir}/${SERVICE_NAME}" "${UNIT_FILE}"', installer)
        self.assertIn('cmp -s "${tmp_dir}/${MANAGER_SERVICE_NAME}" "${MANAGER_UNIT_FILE}"', installer)
        self.assertIn('"${LIB_DIR}/codex_provider.py"', installer)
        self.assertIn('systemctl --user disable --now "${SERVICE_NAME}"', installer)
        self.assertNotIn('"${BIN_DIR}/codex-provider"', installer)

    def test_claudex_only_uses_autocompact_when_claude_supports_it(self):
        installer = INSTALL_SCRIPT.read_text(encoding="utf-8")
        marker = 'cat >"${tmp_dir}/claudex" <<\'EOF\'\n'
        wrapper = installer.split(marker, 1)[1].split("\nEOF\n", 1)[0]

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bin_dir = home / "bin"
            local_bin = home / ".local" / "bin"
            bin_dir.mkdir()
            local_bin.mkdir(parents=True)
            wrapper_path = bin_dir / "claudex"
            wrapper_path.write_text(wrapper, encoding="utf-8")
            wrapper_path.chmod(0o755)
            sync = local_bin / "claude-codex-sync"
            sync.write_text("#!/bin/sh\nprintf 'model\\neffort\\nprovider\\n'\n", encoding="utf-8")
            sync.chmod(0o755)
            claude = bin_dir / "claude"
            claude.write_text(
                "#!/bin/sh\n"
                "if [ \"${1:-}\" = --help ]; then\n"
                "  [ \"${FAKE_AUTOCOMPACT:-0}\" = 1 ] && printf '%s\\n' '  --autocompact <auto|tokens>'\n"
                "  exit 0\n"
                "fi\n"
                "printf '%s\\n' \"$@\"\n",
                encoding="utf-8",
            )
            claude.chmod(0o755)
            environment = os.environ.copy()
            environment.update({"HOME": str(home), "PATH": f"{bin_dir}:{environment['PATH']}"})

            for supported in (False, True):
                with self.subTest(supported=supported):
                    environment["FAKE_AUTOCOMPACT"] = "1" if supported else "0"
                    completed = subprocess.run(
                        [str(wrapper_path)],
                        capture_output=True,
                        text=True,
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    arguments = completed.stdout.splitlines()
                    self.assertEqual("--autocompact" in arguments, supported)
                    self.assertEqual("250k" in arguments, supported)

            yolo_settings = home / ".claude" / "settings.claudex-yolo.json"
            yolo_settings.parent.mkdir()
            yolo_settings.write_text("{}\n", encoding="utf-8")
            environment["CLAUDEX_YOLO"] = "1"
            completed = subprocess.run(
                [str(wrapper_path)],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(str(yolo_settings), completed.stdout.splitlines())

    def test_default_modes_keep_all_skills_and_yolo_uses_dedicated_settings(self):
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
        self.assertIn(
            'YOLO_MINIMAL_SKILLS="agent-reach brainstorming domain-modeling grilling tdd"',
            setup,
        )
        self.assertIn("CLAUDE_SETTINGS_CLAUDEX_YOLO", setup)
        self.assertIn(
            "alias claude-yolo='claude --dangerously-skip-permissions --settings ~/.claude/settings.yolo.json'",
            setup,
        )
        self.assertNotIn("--safe-mode", setup)

    def test_codex_yolo_uses_a_named_config_profile(self):
        setup = SETUP_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'readonly CODEX_CONFIG_YOLO="${CODEX_DIR}/yolo.config.toml"',
            setup,
        )
        self.assertIn(
            "alias codex-yolo='codex --dangerously-bypass-approvals-and-sandbox -p yolo'",
            setup,
        )
        self.assertNotIn(" -c ~/.codex/config.yolo.toml", setup)

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
            codex_skill = home / ".codex" / "skills" / "system-helper" / "SKILL.md"
            codex_skill.parent.mkdir(parents=True)
            codex_skill.write_text("---\nname: system-helper\n---\n", encoding="utf-8")
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
            yolo_config = home / ".codex" / "yolo.config.toml"
            secrets = home / ".config" / "codex" / "secrets.env"
            config_text = config.read_text(encoding="utf-8")
            yolo_config_text = yolo_config.read_text(encoding="utf-8")
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
            for skill_name in self.enabled_skills + self.internal_skills + self.disabled_skills:
                self.assertIn(
                    'path = "'
                    + str(home / ".agents" / "skills" / skill_name / "SKILL.md")
                    + '"\nenabled = true',
                    config_text,
                )
                self.assertIn(
                    'path = "'
                    + str(home / ".claude" / "skills" / skill_name / "SKILL.md")
                    + '"\nenabled = false',
                    config_text,
                )
            self.assertIn(f'path = "{codex_skill}"\nenabled = true', config_text)
            yolo_skill_names = {
                "agent-reach",
                "brainstorming",
                "domain-modeling",
                "grilling",
                "tdd",
            }
            for skill_root in (home / ".agents" / "skills", home / ".claude" / "skills"):
                for skill_name in self.enabled_skills + self.internal_skills + self.disabled_skills:
                    expected = (
                        "true"
                        if skill_root == home / ".agents" / "skills"
                        and skill_name in yolo_skill_names
                        else "false"
                    )
                    self.assertIn(
                        f'path = "{skill_root / skill_name / "SKILL.md"}"\nenabled = {expected}',
                        yolo_config_text,
                    )
            self.assertIn(f'path = "{codex_skill}"\nenabled = false', yolo_config_text)
            managed_yolo_skills = yolo_config_text.split(
                "# >>> scripts disabled Codex skills >>>", 1
            )[1]
            self.assertEqual(managed_yolo_skills.count("enabled = true"), 5)
            self.assertNotIn(secret_value, config_text)
            self.assertIn(secret_value, secrets.read_text(encoding="utf-8"))
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(secrets.stat().st_mode & 0o777, 0o600)
            settings = home / ".claude" / "settings.json"
            settings_data = json.loads(settings.read_text(encoding="utf-8"))
            self.assertNotIn("skillOverrides", settings_data)
            yolo_settings = json.loads(
                (home / ".claude" / "settings.yolo.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {name for name, state in yolo_settings["skillOverrides"].items() if state == "on"},
                yolo_skill_names,
            )
            for skill_name in self.disabled_skills:
                self.assertEqual(yolo_settings["skillOverrides"][skill_name], "off")
            claudex_yolo_settings = json.loads(
                (home / ".claude" / "settings.claudex-yolo.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                claudex_yolo_settings["availableModels"], ["claudex-router[1m]"]
            )
            self.assertEqual(
                claudex_yolo_settings["skillOverrides"]["claude-api"], "off"
            )
            for skill_name in self.disabled_skills:
                self.assertEqual(
                    claudex_yolo_settings["skillOverrides"][skill_name], "off"
                )
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
            self.assertIn(
                "alias claude-yolo='claude --dangerously-skip-permissions --settings ~/.claude/settings.yolo.json'",
                bashrc_text,
            )
            self.assertIn(
                "alias claudex-yolo='CLAUDEX_YOLO=1 claudex'",
                bashrc_text,
            )
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


class CodexAuthSwitchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary_directory.name)
        self.codex_home = self.home / ".codex"
        self.codex_home.mkdir()
        self.config = self.codex_home / "config.toml"
        self.config.write_text(
            'model_provider = "test_route"\n'
            'preferred_auth_method = "apikey"\n'
            'model = "gpt-test"\n\n'
            '[model_providers.test_route]\n'
            'base_url = "https://example.invalid/openai"\n'
            'wire_api = "responses"\n'
            'env_key = "TEST_ROUTE_KEY"\n',
            encoding="utf-8",
        )
        self.yolo_config = self.codex_home / "yolo.config.toml"
        self.yolo_config.write_text(
            'model_provider = "test_route"\n\n'
            '[[skills.config]]\n'
            'path = "/tmp/example/SKILL.md"\n'
            'enabled = true\n',
            encoding="utf-8",
        )
        self.fake_bin = self.home / "bin"
        self.fake_bin.mkdir()
        self.codex_log = self.home / "codex.log"
        fake_codex = self.fake_bin / "codex"
        fake_codex.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_CODEX_LOG\"\n"
            "if [ \"$1 $2\" = 'login status' ]; then\n"
            "  [ \"${FAKE_CHATGPT_LOGIN:-0}\" = 1 ] && printf '%s\\n' 'Logged in using ChatGPT' && exit 0\n"
            "  printf '%s\\n' 'Not logged in'\n"
            "  exit 1\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "HOME": str(self.home),
                "CODEX_HOME": str(self.codex_home),
                "PATH": f"{self.fake_bin}:{self.environment['PATH']}",
                "FAKE_CODEX_LOG": str(self.codex_log),
            }
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_switch(self, *arguments, **environment):
        child_environment = self.environment.copy()
        child_environment.update(environment)
        return subprocess.run(
            ["bash", str(AUTH_SWITCH_SCRIPT), *arguments],
            capture_output=True,
            text=True,
            env=child_environment,
        )

    def test_account_mode_saves_api_provider_and_starts_chatgpt_login(self):
        completed = self.run_switch("account")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        config_text = self.config.read_text(encoding="utf-8")
        self.assertIn('model_provider = "openai"', config_text)
        self.assertIn('preferred_auth_method = "chatgpt"', config_text)
        self.assertIn('forced_login_method = "chatgpt"', config_text)
        self.assertIn('[model_providers.test_route]', config_text)
        self.assertIn('model_provider = "openai"', self.yolo_config.read_text(encoding="utf-8"))
        saved_provider = self.home / ".config" / "codex" / "api-provider"
        self.assertEqual(saved_provider.read_text(encoding="utf-8"), "test_route\n")
        self.assertEqual(saved_provider.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.codex_log.read_text(encoding="utf-8").splitlines(), ["login status", "login"])

    def test_account_mode_reuses_an_existing_chatgpt_login(self):
        completed = self.run_switch("account", FAKE_CHATGPT_LOGIN="1")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.codex_log.read_text(encoding="utf-8").splitlines(), ["login status"])

    def test_api_mode_restores_the_saved_provider(self):
        account_result = self.run_switch("account", FAKE_CHATGPT_LOGIN="1")
        self.assertEqual(account_result.returncode, 0, account_result.stderr)

        completed = self.run_switch("api")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        config_text = self.config.read_text(encoding="utf-8")
        self.assertIn('model_provider = "test_route"', config_text)
        self.assertIn('preferred_auth_method = "apikey"', config_text)
        self.assertNotIn("forced_login_method", config_text)
        self.assertIn('model_provider = "test_route"', self.yolo_config.read_text(encoding="utf-8"))

    def test_status_reports_the_active_mode_without_showing_secrets(self):
        completed = self.run_switch("status")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mode: api", completed.stdout)
        self.assertIn("provider: test_route", completed.stdout)
        self.assertNotIn("TEST_ROUTE_KEY", completed.stdout)

    def test_setup_installs_the_auth_switch_command(self):
        environment = self.environment.copy()
        environment.update(
            {
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
        installed = self.home / ".local" / "bin" / "codex-auth"
        self.assertTrue(installed.is_file())
        self.assertTrue(os.access(installed, os.X_OK))
        self.assertEqual(
            installed.read_text(encoding="utf-8"),
            AUTH_SWITCH_SCRIPT.read_text(encoding="utf-8"),
        )

    def test_setup_preserves_account_mode(self):
        account_result = self.run_switch("account", FAKE_CHATGPT_LOGIN="1")
        self.assertEqual(account_result.returncode, 0, account_result.stderr)
        environment = self.environment.copy()
        environment.update(
            {
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
        self.assertIn('model_provider = "openai"', self.config.read_text(encoding="utf-8"))
        self.assertIn("Codex account mode is active", completed.stdout)


if __name__ == "__main__":
    unittest.main()
