import json
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, call, patch

import main


def fake_client(*responses: str) -> SimpleNamespace:
    queued = iter(responses)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        message = SimpleNamespace(content=next(queued))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        requests=requests,
    )


def judge_json(scores: list[int], feedback: list[str] | None = None) -> str:
    names = (
        "age_appropriateness",
        "story_arc",
        "engagement",
        "category_fit",
        "safety",
        "choice_honored",
    )
    if len(scores) == 5:
        scores = [*scores, 5]
    return json.dumps(
        {
            "verdict": "pass",
            "scores": dict(zip(names, scores)),
            "feedback": feedback or [],
        }
    )


def passing_result() -> main.JudgeResult:
    return main.JudgeResult(
        "pass",
        {
            "age_appropriateness": 5,
            "story_arc": 5,
            "engagement": 5,
            "category_fit": 5,
            "safety": 5,
            "choice_honored": 5,
        },
        [],
    )


class StoryPipelineTests(unittest.TestCase):
    def test_router_uses_safe_fallback_for_unexpected_output(self):
        self.assertEqual(
            main.route_request(fake_client("other"), "a tale"), "calm bedtime"
        )

    def test_age_prompt_accepts_valid_age_and_defaults(self):
        with patch("builtins.input", return_value=""):
            self.assertEqual(main.prompt_for_age(), 7)
        with patch("builtins.input", return_value="10"):
            self.assertEqual(main.prompt_for_age(), 10)
        with (
            patch("builtins.input", side_effect=["eleven", "still invalid"]),
            patch("builtins.print"),
        ):
            self.assertEqual(main.prompt_for_age(), 7)

    def test_judge_prompt_uses_specific_listener_age(self):
        client = fake_client(judge_json([5, 5, 5, 5, 5, 5]))
        main.judge_story(client, "a tale", "adventure", "A story.", 6)

        prompt = client.requests[0]["messages"][1]["content"]
        self.assertIn("for a 6-year-old", prompt)
        self.assertNotIn("ages 5-10", prompt)

    def test_judge_derives_failure_from_scores_not_claimed_verdict(self):
        result = main.judge_story(
            fake_client(judge_json([5, 3, 5, 5, 5], ["Make the resolution clearer."])),
            "a tale",
            "adventure",
            "Once upon a time...",
            7,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.scores["story_arc"], 3)

    def test_judge_requires_choice_to_be_honored(self):
        result = main.judge_story(
            fake_client(
                judge_json(
                    [5, 5, 5, 5, 5, 3],
                    ["Make the chosen path cause the resolution."],
                )
            ),
            "a tale",
            "adventure",
            "Once upon a time...",
            7,
            "Follow the fireflies.",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.scores["choice_honored"], 3)

    @patch("main.judge_story")
    @patch("main.generate_ending")
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
                "choice_honored": 3,
            },
            ["Improve it."],
        )
        judge.side_effect = [failing, failing, failing]

        story, _result, revisions = main.create_judged_story(
            object(), "a tale", "adventure", "opening", "take the bridge", 7
        )

        self.assertEqual(revisions, 2)
        self.assertEqual(story, "opening\n\ndraft")
        self.assertEqual(generate.call_count, 3)

    @patch("main.judge_story")
    @patch("main.generate_ending")
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
                "choice_honored": 5,
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
                "choice_honored": 4,
            },
            [],
        )
        judge.side_effect = [high_failure, passing]

        story, result, revisions = main.create_judged_story(
            object(), "a tale", "adventure", "opening", "take the bridge", 7
        )

        self.assertEqual(story, "opening\n\npassing rewrite")
        self.assertTrue(result.passed)
        self.assertEqual(revisions, 1)

    def test_interactive_change_is_rewritten_and_rechecked(self):
        passing = passing_result()
        with (
            patch("main.make_client", return_value=object()),
            patch("main.prompt_for_age", return_value=7),
            patch("main.route_request", return_value="calm bedtime"),
            patch(
                "main.generate_opening",
                return_value="opening\nDECISION: Which path should Alice take?",
            ),
            patch("main.generate_choices", return_value=["Path one", "Path two"]),
            patch(
                "main.create_judged_story",
                return_value=("opening\n\nending", passing, 0),
            ),
            patch("main.generate_story", return_value="friendlier rewrite") as rewrite,
            patch("main.judge_story", return_value=passing) as recheck,
            patch(
                "builtins.input",
                side_effect=[main.EXAMPLE_REQUEST, "1", "make Bob fluffier", ""],
            ),
            patch("builtins.print"),
        ):
            main.run()

        self.assertEqual(rewrite.call_count, 1)
        self.assertIn(
            "make Bob fluffier",
            rewrite.call_args.kwargs["revision_instructions"],
        )
        recheck.assert_called_once_with(
            ANY,
            main.EXAMPLE_REQUEST,
            "calm bedtime",
            "friendlier rewrite",
            7,
            "Path one",
        )

    def test_exit_responses_finish_without_rewrite(self):
        passing = passing_result()
        for response in ("no", "N", "done", "quit", "exit", ""):
            with self.subTest(response=response):
                with (
                    patch("main.make_client", return_value=object()),
                    patch("main.prompt_for_age", return_value=7),
                    patch("main.route_request", return_value="calm bedtime"),
                    patch(
                        "main.generate_opening",
                        return_value="opening\nDECISION: Which path?",
                    ),
                    patch(
                        "main.generate_choices", return_value=["Path one", "Path two"]
                    ),
                    patch(
                        "main.create_judged_story",
                        return_value=("opening\n\nending", passing, 0),
                    ),
                    patch("main.generate_story") as rewrite,
                    patch(
                        "builtins.input",
                        side_effect=[main.EXAMPLE_REQUEST, "1", response],
                    ),
                    patch("builtins.print") as output,
                ):
                    main.run()

                rewrite.assert_not_called()
                output.assert_any_call("Good night!")

    def test_safety_override_acknowledgment_prints_only_for_trigger(self):
        passing = passing_result()
        for request, should_acknowledge in (
            ("A fox story with a sad ending", True),
            ("A sad story about a little robot", True),
            ("A cozy fox story under the stars", False),
        ):
            with self.subTest(request=request):
                with (
                    patch("main.make_client", return_value=object()),
                    patch("main.prompt_for_age", return_value=7),
                    patch("main.route_request", return_value="calm bedtime"),
                    patch(
                        "main.generate_opening",
                        return_value="opening\nDECISION: Which path?",
                    ),
                    patch(
                        "main.generate_choices", return_value=["Path one", "Path two"]
                    ),
                    patch(
                        "main.create_judged_story",
                        return_value=("opening\n\nending", passing, 0),
                    ),
                    patch("builtins.input", side_effect=[request, "1", ""]),
                    patch("builtins.print") as output,
                ):
                    main.run()

                acknowledgment = call(main.SAFETY_ACKNOWLEDGMENT)
                if should_acknowledge:
                    routed = call("\nRouted as: calm bedtime")
                    routed_index = output.call_args_list.index(routed)
                    self.assertEqual(
                        output.call_args_list[routed_index + 1], acknowledgment
                    )
                    self.assertEqual(output.call_args_list.count(acknowledgment), 1)
                else:
                    self.assertNotIn(acknowledgment, output.call_args_list)

    def test_choice_parsing_requires_exactly_two_non_empty_choices(self):
        valid = main.generate_choices(
            fake_client('{"choices": [" Follow the fireflies. ", "Ask the owl."]}'),
            "opening\nDECISION: Which path?",
            "adventure",
            7,
        )
        self.assertEqual(valid, ["Follow the fireflies.", "Ask the owl."])

        for malformed in (
            "not json",
            '{"choices": ["Only one."]}',
            '{"choices": ["", "Ask the owl."]}',
            '{"choices": ["Same.", "same."]}',
        ):
            with self.subTest(malformed=malformed):
                choices = main.generate_choices(
                    fake_client(malformed),
                    "opening\nDECISION: Which path?",
                    "adventure",
                    7,
                )
                self.assertEqual(choices, list(main.DEFAULT_CHOICES))

    def test_choice_prompt_uses_placeholders_not_concrete_examples(self):
        with patch(
            "main.call_model",
            return_value='{"choices": ["Climb.", "Wait."]}',
        ) as model:
            main.generate_choices(object(), "An opening.", "adventure", 7)

        prompt = model.call_args.args[1][1]["content"]
        self.assertIn("<first choice>", prompt)
        self.assertNotIn("Follow the lights", prompt)
        self.assertNotIn("Ask the owl", prompt)

    def test_change_hint_uses_available_story_elements(self):
        with patch("main.random.choice", side_effect=lambda choices: choices[0]):
            creature_hint = main.build_change_hint("Mina", "A dragon waved.")
            hero_hint = main.build_change_hint("Mina", "The path glowed.")

        self.assertIn("dragon", creature_hint)
        self.assertIn("Mina", hero_hint)
        self.assertNotIn("dragon", hero_hint)
        self.assertNotRegex(hero_hint, r"\{.+\}")

    def test_change_hint_varies_for_same_opening(self):
        main.random.seed(7)
        hints = {
            main.build_change_hint("Mina", "Mina met a rabbit.")
            for _ in range(12)
        }
        self.assertGreater(len(hints), 1)

    def test_empty_and_invalid_choice_default_to_first(self):
        choices = ["Follow the fireflies.", "Ask the owl."]
        with (
            patch("builtins.input", return_value=""),
            patch("builtins.print"),
        ):
            self.assertEqual(
                main.prompt_for_choice("Alice", choices),
                "Follow the fireflies.",
            )

        with (
            patch("builtins.input", side_effect=["three", "still not a number"]),
            patch("builtins.print"),
        ):
            self.assertEqual(
                main.prompt_for_choice("Alice", choices),
                "Follow the fireflies.",
            )

    def test_finish_words_exit_at_choice(self):
        for response in ("no", "N", "done", "quit", "exit"):
            with self.subTest(response=response):
                with (
                    patch("builtins.input", return_value=f"  {response}  "),
                    patch("builtins.print"),
                ):
                    self.assertIsNone(
                        main.prompt_for_choice("Alice", ["Path one", "Path two"])
                    )

    @patch("main.generate_opening")
    @patch("main.judge_story")
    @patch("main.generate_ending")
    def test_choice_failure_revises_ending_not_opening(
        self, generate_ending, judge, generate_opening
    ):
        generate_ending.side_effect = ["ending ignores choice", "ending honors choice"]
        choice_failure = main.JudgeResult(
            "fail",
            {
                "age_appropriateness": 5,
                "story_arc": 5,
                "engagement": 5,
                "category_fit": 5,
                "safety": 5,
                "choice_honored": 2,
            },
            ["Make the firefly path directly solve the problem."],
        )
        judge.side_effect = [choice_failure, passing_result()]

        story, result, revisions = main.create_judged_story(
            object(),
            "an Alice adventure",
            "adventure",
            "fixed opening",
            "Follow the fireflies.",
            7,
        )

        self.assertEqual(story, "fixed opening\n\nending honors choice")
        self.assertTrue(result.passed)
        self.assertEqual(revisions, 1)
        self.assertEqual(generate_ending.call_count, 2)
        self.assertTrue(
            all(
                call_args.args[1] == "fixed opening"
                for call_args in generate_ending.call_args_list
            )
        )
        generate_opening.assert_not_called()

    def test_decision_metadata_is_never_displayed(self):
        with (
            patch("main.make_client", return_value=object()),
            patch("main.prompt_for_age", return_value=7),
            patch("main.route_request", return_value="adventure"),
            patch(
                "main.generate_opening",
                return_value=(
                    "Alice reached the moonlit fork.\n"
                    "metadata:\n"
                    "DECISION: Should Alice follow the lights or ask the owl?"
                ),
            ),
            patch("main.generate_choices", return_value=["Follow lights", "Ask owl"]),
            patch("main.create_judged_story") as create_story,
            patch("builtins.input", side_effect=["A girl named Alice", "exit"]),
            patch("builtins.print") as output,
        ):
            main.run()

        displayed = "\n".join(
            str(argument)
            for printed_call in output.call_args_list
            for argument in printed_call.args
        )
        self.assertNotIn("DECISION:", displayed)
        self.assertNotIn("metadata:", displayed)
        self.assertIn("Alice reached the moonlit fork.", displayed)
        create_story.assert_not_called()

    def test_hero_metadata_names_the_choice_prompt(self):
        with (
            patch("main.make_client", return_value=object()),
            patch("main.prompt_for_age", return_value=7),
            patch("main.route_request", return_value="adventure"),
            patch(
                "main.generate_opening",
                return_value="An opening.\nDECISION: Which way?\nHERO: Finn",
            ),
            patch("main.generate_choices", return_value=["Go left", "Go right"]),
            patch("main.create_judged_story"),
            patch("builtins.input", side_effect=["A forest story", "exit"]),
            patch("builtins.print") as output,
        ):
            main.run()

        output.assert_any_call(
            "\nWhat should Finn do?\n  1. Go left\n  2. Go right"
        )


if __name__ == "__main__":
    unittest.main()
