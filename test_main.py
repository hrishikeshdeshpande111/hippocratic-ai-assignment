import json
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import main


def fake_client(*responses: str) -> SimpleNamespace:
    queued = iter(responses)

    def create(**_kwargs):
        message = SimpleNamespace(content=next(queued))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def judge_json(scores: list[int], feedback: list[str] | None = None) -> str:
    names = (
        "age_appropriateness",
        "story_arc",
        "engagement",
        "category_fit",
        "safety",
    )
    return json.dumps(
        {
            "verdict": "pass",
            "scores": dict(zip(names, scores)),
            "feedback": feedback or [],
        }
    )


class StoryPipelineTests(unittest.TestCase):
    def test_router_uses_safe_fallback_for_unexpected_output(self):
        self.assertEqual(main.route_request(fake_client("other"), "a tale"), "calm bedtime")

    def test_judge_derives_failure_from_scores_not_claimed_verdict(self):
        result = main.judge_story(
            fake_client(judge_json([5, 3, 5, 5, 5], ["Make the resolution clearer."])),
            "a tale",
            "adventure",
            "Once upon a time...",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.scores["story_arc"], 3)

    @patch("main.judge_story")
    @patch("main.generate_story")
    def test_revision_loop_is_capped_at_two(self, generate, judge):
        generate.side_effect = ["draft", "rewrite one", "rewrite two"]
        failing = main.JudgeResult(
            "fail",
            {
                "age_appropriateness": 3,
                "story_arc": 3,
                "engagement": 3,
                "category_fit": 3,
                "safety": 3,
            },
            ["Improve it."],
        )
        judge.side_effect = [failing, failing, failing]

        story, _result, revisions = main.create_judged_story(
            object(), "a tale", "adventure"
        )

        self.assertEqual(revisions, 2)
        self.assertEqual(story, "draft")
        self.assertEqual(generate.call_count, 3)

    @patch("main.judge_story")
    @patch("main.generate_story")
    def test_passing_rewrite_beats_higher_scoring_failure(self, generate, judge):
        generate.side_effect = ["high total failure", "passing rewrite"]
        high_failure = main.JudgeResult(
            "fail",
            {
                "age_appropriateness": 5,
                "story_arc": 3,
                "engagement": 5,
                "category_fit": 5,
                "safety": 5,
            },
            ["Clarify the arc."],
        )
        passing = main.JudgeResult(
            "pass",
            {
                "age_appropriateness": 4,
                "story_arc": 4,
                "engagement": 4,
                "category_fit": 4,
                "safety": 4,
            },
            [],
        )
        judge.side_effect = [high_failure, passing]

        story, result, revisions = main.create_judged_story(
            object(), "a tale", "adventure"
        )

        self.assertEqual(story, "passing rewrite")
        self.assertTrue(result.passed)
        self.assertEqual(revisions, 1)

    def test_interactive_change_is_rewritten_and_rechecked(self):
        passing = main.JudgeResult(
            "pass",
            {
                "age_appropriateness": 5,
                "story_arc": 5,
                "engagement": 5,
                "category_fit": 5,
                "safety": 5,
            },
            [],
        )
        with (
            patch("main.make_client", return_value=object()),
            patch("main.route_request", return_value="calm bedtime"),
            patch("main.create_judged_story", return_value=("draft", passing, 0)),
            patch("main.generate_story", return_value="friendlier rewrite") as rewrite,
            patch("main.judge_story", return_value=passing) as recheck,
            patch("builtins.input", side_effect=[main.EXAMPLE_REQUEST, "make Bob fluffier", ""]),
            patch("builtins.print"),
        ):
            main.run()

        self.assertEqual(rewrite.call_count, 1)
        self.assertIn("make Bob fluffier", rewrite.call_args.kwargs["revision_instructions"])
        recheck.assert_called_once_with(
            ANY,
            main.EXAMPLE_REQUEST,
            "calm bedtime",
            "friendlier rewrite",
        )


if __name__ == "__main__":
    unittest.main()
