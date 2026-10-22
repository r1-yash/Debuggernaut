import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from swe_agent.agent.fixer import ExplorationStep, propose_fix


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


if __name__ == "__main__":
    unittest.main()
