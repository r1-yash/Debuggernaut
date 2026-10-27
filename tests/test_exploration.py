import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swe_agent.llm import MalformedProviderResponse, OpenAICompatibleProvider


def _tool_call(
    call_id: str = "call-1",
    name: str = "read_repository_file",
    arguments: str = '{"file_path": "a.py"}',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *, tool_calls: list[SimpleNamespace] | None = None, content: str | None = None,
) -> SimpleNamespace:
    calls = tool_calls or []

    def model_dump(**_: object) -> dict:
        message: dict = {"role": "assistant"}
        if content is not None:
            message["content"] = content
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ]
        return message

    message = SimpleNamespace(
        tool_calls=calls, content=content, model_dump=model_dump
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _provider(rounds: int) -> OpenAICompatibleProvider:
    with patch("openai.OpenAI"):
        return OpenAICompatibleProvider(
            "qwen",
            "test-model",
            "test-key",
            "https://example.test/v1",
            exploration_rounds=rounds,
        )


def _fake_client(responses: list[SimpleNamespace], requests: list[dict]) -> SimpleNamespace:
    def create(**kwargs: object) -> SimpleNamespace:
        requests.append(
            {
                "tool_choice": kwargs.get("tool_choice"),
                "messages": [dict(message) for message in kwargs["messages"]],
                "tool_count": len(kwargs.get("tools") or []),
            }
        )
        return responses[len(requests) - 1]

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _read_file_tool(executed: list[str]):
    def read_repository_file(file_path: str) -> str:
        """Read a repository file."""
        executed.append(file_path)
        return "file contents"

    return read_repository_file


class ExplorationRoundTests(unittest.TestCase):
    def test_always_requesting_tools_gets_a_forced_final_round(self) -> None:
        provider = _provider(3)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(tool_calls=[_tool_call()]),
            _response(tool_calls=[_tool_call(call_id="call-2")]),
            _response(content="Final answer from the explored files."),
        ]
        provider.client = _fake_client(responses, requests)

        _, response, rounds = provider.explore(
            "system", "prompt", [_read_file_tool(executed)]
        )

        self.assertEqual(rounds, 3)
        self.assertEqual(len(requests), 3)
        self.assertIsNone(requests[0]["tool_choice"])
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual(requests[-1]["tool_count"], 1)
        self.assertEqual(executed, ["a.py", "a.py"])
        self.assertTrue(
            any(
                isinstance(message.get("content"), str)
                and "Do not call any more tools" in message["content"]
                for message in requests[-1]["messages"]
            )
        )
        self.assertEqual(
            response.choices[0].message.content,
            "Final answer from the explored files.",
        )

    def test_model_that_finishes_in_two_rounds_returns_early(self) -> None:
        provider = _provider(10)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(tool_calls=[_tool_call()]),
            _response(content="Done."),
        ]
        provider.client = _fake_client(responses, requests)

        _, _, rounds = provider.explore(
            "system", "prompt", [_read_file_tool(executed)]
        )

        self.assertEqual(rounds, 2)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request["tool_choice"] is None for request in requests))
        self.assertEqual(executed, ["a.py"])

    def test_forced_round_without_text_raises_clear_error(self) -> None:
        provider = _provider(2)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(tool_calls=[_tool_call()]),
            _response(content=None),
        ]
        provider.client = _fake_client(responses, requests)

        with self.assertRaisesRegex(
            MalformedProviderResponse,
            "no final answer after tool calls were disabled",
        ):
            provider.explore("system", "prompt", [_read_file_tool(executed)])

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual(executed, ["a.py"])

    def test_forced_round_text_with_tool_calls_is_sanitized(self) -> None:
        provider = _provider(2)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(tool_calls=[_tool_call()]),
            _response(
                content="Final answer.",
                tool_calls=[_tool_call(call_id="ignored")],
            ),
        ]
        provider.client = _fake_client(responses, requests)

        history, _, rounds = provider.explore(
            "system", "prompt", [_read_file_tool(executed)]
        )

        self.assertEqual(rounds, 2)
        self.assertEqual(history[-1], {"role": "assistant", "content": "Final answer."})
        self.assertEqual(set(history[-1]), {"role", "content"})
        self.assertEqual(executed, ["a.py"])
        self.assertFalse(
            any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == "ignored"
                for message in history
            )
        )

    def test_forced_round_text_only_is_appended_unchanged(self) -> None:
        provider = _provider(2)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(tool_calls=[_tool_call()]),
            _response(content="Final answer."),
        ]
        provider.client = _fake_client(responses, requests)

        history, _, _ = provider.explore(
            "system", "prompt", [_read_file_tool(executed)]
        )

        self.assertEqual(history[-1], {"role": "assistant", "content": "Final answer."})

    def test_non_final_tool_calls_keep_matching_tool_replies(self) -> None:
        provider = _provider(3)
        executed: list[str] = []
        requests: list[dict] = []
        responses = [
            _response(
                tool_calls=[
                    _tool_call(call_id="call-1"),
                    _tool_call(call_id="call-2"),
                ]
            ),
            _response(content="Done."),
        ]
        provider.client = _fake_client(responses, requests)

        history, _, rounds = provider.explore(
            "system", "prompt", [_read_file_tool(executed)]
        )

        self.assertEqual(rounds, 2)
        self.assertEqual(
            [call["id"] for call in history[-4]["tool_calls"]],
            ["call-1", "call-2"],
        )
        self.assertEqual(history[-3]["role"], "tool")
        self.assertEqual(history[-3]["tool_call_id"], "call-1")
        self.assertEqual(history[-2]["tool_call_id"], "call-2")
        self.assertEqual(history[-1], {"role": "assistant", "content": "Done."})
        self.assertEqual(executed, ["a.py", "a.py"])
