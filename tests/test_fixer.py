import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from swe_agent.agent.fixer import (
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    ExplorationStep,
    _tool_result_char_limit,
    list_directory,
    propose_fix,
    read_file,
    search_code,
)
from swe_agent.agent.guardrails import RejectedProposalError


class FixerTests(unittest.TestCase):
    def test_proposes_fix_from_mocked_genai_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "parser.py").write_text("def parse():\n    return None\n")
            client = Mock()
            chat = client.chats.create.return_value
            chat.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "parser.py",
                "new_content": "def parse():\n    return {}\n",
                "reasoning": "Return an empty mapping instead of None.",
            }

            result = propose_fix(
                str(repository),
                "Parser returns None",
                "Empty inputs should return an empty mapping.",
                client,
            )

            self.assertEqual(result.file_path, "parser.py")
            self.assertEqual(result.original_content, "def parse():\n    return None\n")
            self.assertEqual(result.new_content, "def parse():\n    return {}\n")
            self.assertEqual(
                result.reasoning, "Return an empty mapping instead of None."
            )
            config = client.models.generate_content.call_args.kwargs["config"]
            self.assertIsNone(config.tools)
            self.assertEqual(config.response_mime_type, "application/json")
            chat_config = client.chats.create.call_args.kwargs["config"]
            self.assertEqual(len(chat_config.tools), 3)
            chat.send_message.assert_called_once_with(
                "Issue title: Parser returns None\n\n"
                "Issue body:\nEmpty inputs should return an empty mapping."
            )

    def test_raises_when_tool_call_cap_has_no_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            client = Mock()
            chat = client.chats.create.return_value
            chat.get_history.return_value = []
            client.models.generate_content.return_value.parsed = None
            client.models.generate_content.return_value.text = None

            with self.assertRaisesRegex(ValueError, "parseable structured response"):
                propose_fix(str(repository), "Needs a fix", "Details", client)

    def test_records_exploration_steps_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "parser.py").write_text("def parse():\n    return None\n")
            client = Mock()
            chat = client.chats.create.return_value
            chat.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "parser.py",
                "new_content": "def parse():\n    return {}\n",
                "reasoning": "Return an empty mapping instead of None.",
            }
            steps: list[ExplorationStep] = []

            def send_message(prompt: str) -> Mock:
                tools = client.chats.create.call_args.kwargs["config"].tools
                tools[0]("parser.py")
                tools[1](".")
                tools[2]("def parse")
                return Mock()

            chat.send_message.side_effect = send_message

            propose_fix(
                str(repository), "Parser returns None", "Details", client,
                exploration_steps=steps,
            )

            self.assertEqual(
                [step.tool for step in steps],
                ["read_file", "list_directory", "search_code"],
            )
            self.assertTrue(all(step.ok for step in steps))
            self.assertGreater(steps[0].result_size, 0)
            self.assertEqual(steps[0].result_unit, "characters")
            self.assertEqual(steps[1].result_unit, "entries")
            self.assertEqual(steps[2].result_unit, "lines")
            self.assertEqual(steps[0].arguments, {"path": "parser.py"})
            self.assertEqual(steps[2].arguments, {"query": "def parse"})

    def test_records_failed_exploration_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "parser.py").write_text("def parse():\n    return None\n")
            client = Mock()
            chat = client.chats.create.return_value
            chat.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "parser.py",
                "new_content": "",
                "reasoning": "No-op.",
            }
            steps: list[ExplorationStep] = []

            def send_message(prompt: str) -> Mock:
                tools = client.chats.create.call_args.kwargs["config"].tools
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    tools[0]("missing.py")
                return Mock()

            chat.send_message.side_effect = send_message

            propose_fix(
                str(repository), "Parser returns None", "Details", client,
                exploration_steps=steps,
            )

            self.assertEqual(len(steps), 1)
            self.assertFalse(steps[0].ok)
            self.assertEqual(steps[0].result_size, 0)
            self.assertIn("missing.py", steps[0].error)

    def test_rejects_test_file_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "tests").mkdir()
            (repository / "tests" / "test_parser.py").write_text("def test_x():\n")
            client = Mock()
            client.chats.create.return_value.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "tests/test_parser.py",
                "new_content": "def test_x():\n    assert False\n",
                "reasoning": "Change the test.",
            }

            with self.assertRaises(RejectedProposalError) as context:
                propose_fix(
                    str(repository), "Parser returns None", "Details", client,
                    test_command=["pytest"],
                )

            self.assertEqual(context.exception.kind, "test_file")
            self.assertEqual(
                context.exception.file_path, "tests/test_parser.py"
            )
            self.assertEqual(
                (repository / "tests" / "test_parser.py").read_text(),
                "def test_x():\n",
            )

    def test_rejects_test_patch_file_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            fixtures = repository / "fixtures"
            fixtures.mkdir()
            (fixtures / "helper.py").write_text("VALUE = 1\n")
            client = Mock()
            client.chats.create.return_value.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "fixtures/helper.py",
                "new_content": "VALUE = 2\n",
                "reasoning": "Change the fixture.",
            }

            with self.assertRaises(RejectedProposalError) as context:
                propose_fix(
                    str(repository), "Parser returns None", "Details", client,
                    protected_paths=["fixtures/helper.py"],
                )

            self.assertEqual(context.exception.kind, "test_patch_file")
            self.assertIn("benchmark test patch", context.exception.reason)
            self.assertEqual(
                (fixtures / "helper.py").read_text(), "VALUE = 1\n"
            )

    def test_rejects_nonexistent_path_as_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            client = Mock()
            client.chats.create.return_value.get_history.return_value = []
            client.models.generate_content.return_value.parsed = {
                "file_path": "missing.py",
                "new_content": "",
                "reasoning": "Guess.",
            }

            with self.assertRaises(RejectedProposalError) as context:
                propose_fix(
                    str(repository), "Parser returns None", "Details", client
                )

            self.assertEqual(context.exception.kind, "nonexistent_path")
            self.assertEqual(context.exception.file_path, "missing.py")
            self.assertIn("does not exist", context.exception.reason)

    def test_reading_test_files_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "tests").mkdir()
            test_file = repository / "tests" / "test_parser.py"
            test_file.write_text("def test_parse():\n    assert True\n")

            content = read_file(str(repository), "tests/test_parser.py")

            self.assertIn("test_parse", content)

    def test_read_file_truncates_to_char_budget_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "big.py").write_text("x" * 100)

            content = read_file(str(repository), "big.py", _max_chars=10)

            self.assertEqual(
                content, "x" * 10 + "\n[truncated: 90 more chars]"
            )
            self.assertEqual(
                read_file(str(repository), "big.py"), "x" * 100
            )

    def test_search_code_caps_a_single_huge_match_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "bundle.js").write_text(
                "needle " + "y" * 5_000 + "\n"
            )

            matches = search_code(
                str(repository), "needle", _max_chars=200
            )

            self.assertEqual(len(matches), 2)
            self.assertEqual(len(matches[0]), 200)
            self.assertIn("bundle.js", matches[0])
            marker = re.fullmatch(
                r"\[truncated: (\d+) more chars\]", matches[1]
            )
            self.assertIsNotNone(marker)
            self.assertGreater(int(marker.group(1)), 4_000)

    def test_search_code_marks_the_silent_line_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "many.py").write_text(
                "\n".join("needle" for _ in range(60)) + "\n"
            )

            matches = search_code(str(repository), "needle")

            self.assertEqual(len(matches), 51)
            self.assertRegex(
                matches[-1], r"^\[truncated: (\d+) more chars\]$"
            )

    def test_list_directory_truncates_to_char_budget_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            for number in range(30):
                (repository / f"file_{number:02d}.py").touch()

            entries = list_directory(str(repository), _max_chars=25)

            self.assertEqual(entries[:2], ["file_00.py", "file_01.py"])
            self.assertEqual(len(entries), 3)
            self.assertRegex(
                entries[-1], r"^\[truncated: (\d+) more chars\]$"
            )

    def test_tool_result_budget_is_ten_percent_of_input_limit(self) -> None:
        self.assertEqual(
            _tool_result_char_limit(
                SimpleNamespace(max_input_tokens=1_000_000)
            ),
            200_000,
        )
        self.assertEqual(
            _tool_result_char_limit(SimpleNamespace(max_input_tokens=None)),
            DEFAULT_TOOL_RESULT_MAX_CHARS,
        )


if __name__ == "__main__":
    unittest.main()
