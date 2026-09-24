import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from swe_agent.agent.fixer import propose_fix


class FixerTests(unittest.TestCase):
    def test_proposes_fix_from_mocked_genai_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "parser.py").write_text("def parse():\n    return None\n")
            client = Mock()
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
            self.assertEqual(len(config.tools), 3)

    def test_raises_when_tool_call_cap_has_no_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            client = Mock()
            client.models.generate_content.return_value.parsed = None
            client.models.generate_content.return_value.text = None

            with self.assertRaisesRegex(ValueError, "parseable structured response"):
                propose_fix(str(repository), "Needs a fix", "Details", client)


if __name__ == "__main__":
    unittest.main()
