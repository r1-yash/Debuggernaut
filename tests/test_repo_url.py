import unittest

from swe_agent.ingestion.repo_url import parse_repository_url


class ParseRepositoryUrlTests(unittest.TestCase):
    def test_extracts_owner_and_repository(self) -> None:
        self.assertEqual(
            parse_repository_url("https://github.com/openai/swe-agent"),
            "openai/swe-agent",
        )

    def test_accepts_common_pasted_url_variants(self) -> None:
        for url in (
            "github.com/openai/swe-agent/",
            "https://www.github.com/openai/swe-agent.git",
            "https://github.com/openai/swe-agent?tab=readme",
        ):
            with self.subTest(url=url):
                self.assertEqual(parse_repository_url(url), "openai/swe-agent")

    def test_rejects_non_repository_urls(self) -> None:
        with self.assertRaises(ValueError):
            parse_repository_url("https://gitlab.com/openai/swe-agent")
