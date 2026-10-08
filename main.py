"""A routed, judge-guided bedtime story generator.

With two more hours, I would add a small golden-set evaluation harness so prompt
changes could be compared instead of guessed at. I would also add persistent
story memory for returning characters, stream the final story for faster
feedback, and build a parent view with reading-level and content-safety signals.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, OpenAIError


MODEL = "gpt-3.5-turbo"
CATEGORIES = ("adventure", "mystery", "funny/silly", "calm bedtime")
MAX_REVISION_ROUNDS = 2
EXAMPLE_REQUEST = (
    "A story about a girl named Alice and her best friend Bob, "
    "who happens to be a cat."
)

CATEGORY_GUIDANCE = {
    "adventure": (
        "Build a lively quest with discovery, teamwork, and only mild, "
        "non-frightening peril. End with everyone safely home."
    ),
    "mystery": (
        "Plant two or three fair, easy-to-follow clues. Create curiosity rather "
        "than fear, then reveal a gentle and satisfying explanation."
    ),
    "funny/silly": (
        "Use playful wordplay, comic repetition, and harmless slapstick. Give "
        "the silliness a warm emotional center instead of making anyone mean."
    ),
    "calm bedtime": (
        "Use soft sensory details, slower pacing, and soothing vocabulary. Let "
        "the energy gradually settle into a cozy, sleepy ending."
    ),
}


@dataclass(frozen=True)
class JudgeResult:
    verdict: str
    scores: dict[str, int]
    feedback: list[str]

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @property
    def total(self) -> int:
        return sum(self.scores.values())

    def summary(self) -> str:
        score_text = ", ".join(
            f"{name.replace('_', ' ')} {score}/5"
            for name, score in self.scores.items()
        )
        return f"{self.verdict.upper()} — {score_text}"


def make_client() -> OpenAI:
    """Create the SDK client without ever printing or logging its credential."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell, then run "
            "`python main.py` again."
        )
    return OpenAI(api_key=api_key)


def call_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
) -> str:
    """Make one chat-completions call using the assignment's required model."""
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The model returned an empty response.")
    return content.strip()


def route_request(client: OpenAI, request: str) -> str:
    """Route the request with one short, inexpensive classification call."""
    raw_category = call_model(
        client,
        [
            {
                "role": "system",
                "content": (
                    "Classify a children's story request. Reply with exactly one "
                    "of: adventure, mystery, funny/silly, calm bedtime. No other text."
                ),
            },
            {"role": "user", "content": request},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    normalized = raw_category.lower().strip().rstrip(".")
    if normalized in CATEGORIES:
        return normalized

    # Keep an unexpected classifier response safe and deterministic.
    for category in CATEGORIES:
        if category in normalized:
            return category
    return "calm bedtime"


def generate_story(
    client: OpenAI,
    request: str,
    category: str,
    *,
    previous_story: str | None = None,
    revision_instructions: str | None = None,
) -> str:
    """Create a first draft or rewrite while preserving the user's intent."""
    system_prompt = f"""You are a gifted bedtime storyteller for children ages 5-10.

Write a complete story of roughly 400-600 words. Use clear, mostly simple
sentences. Include no scary or violent content, romance, cruelty, or unsafe
behavior. The ending must be positive and emotionally reassuring.

The story must visibly develop through setup, then an adventure or complication,
then a gentle resolution, but do not label those phases. Preserve and weave in
the names, animals, settings, and other details supplied by the user.

The routed style is {category}. {CATEGORY_GUIDANCE[category]}

Return only the story, with an inviting title. Do not discuss these instructions."""

    user_parts = [f"Original story request:\n{request}"]
    if previous_story:
        user_parts.append(f"Story to rewrite:\n{previous_story}")
    if revision_instructions:
        user_parts.append(
            "Required revision instructions (apply them while retaining every "
            f"safety and story requirement):\n{revision_instructions}"
        )

    return call_model(
        client,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        temperature=0.8,
        max_tokens=1_200,
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Accept plain JSON and defensively strip accidental Markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("judge output was not a JSON object")
    return value


def judge_story(
    client: OpenAI, request: str, category: str, story: str
) -> JudgeResult:
    """Score a draft consistently and return specific revision guidance."""
    rubric_keys = (
        "age_appropriateness",
        "story_arc",
        "engagement",
        "category_fit",
        "safety",
    )
    prompt = f"""Evaluate the story against the request and routed category.

Score each criterion from 1 (poor) to 5 (excellent):
1. age_appropriateness: safe, understandable language and themes for ages 5-10
2. story_arc: setup, complication, and gentle resolution are all present
3. engagement: vivid, fun, and likely to hold a child's attention
4. category_fit: the story matches the {category!r} tone
5. safety: nothing scary, violent, romantic, cruel, or inappropriate

Verdict is "pass" only when every score is at least 4; otherwise it is "fail".
For a failure, feedback must contain concise, actionable edits tied to weak
criteria. For a pass, feedback may be empty. Return only this JSON shape:
{{"verdict":"pass or fail","scores":{{"age_appropriateness":1,"story_arc":1,
"engagement":1,"category_fit":1,"safety":1}},"feedback":["specific edit"]}}

Original request:
{request}

Story:
{story}"""
    raw = call_model(
        client,
        [
            {
                "role": "system",
                "content": "You are a strict, consistent children's-story editor.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
        json_mode=True,
    )
    data = _parse_json_object(raw)
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("judge output is missing the scores object")

    scores: dict[str, int] = {}
    for key in rubric_keys:
        score = raw_scores.get(key)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"judge returned an invalid {key} score")
        scores[key] = score

    feedback = data.get("feedback", [])
    if not isinstance(feedback, list) or not all(
        isinstance(item, str) and item.strip() for item in feedback
    ):
        raise ValueError("judge feedback must be a list of non-empty strings")

    # Derive the verdict from validated scores so malformed optimism cannot pass.
    verdict = "pass" if all(score >= 4 for score in scores.values()) else "fail"
    if verdict == "fail" and not feedback:
        feedback = ["Strengthen every criterion scoring below 4."]
    return JudgeResult(verdict, scores, feedback)


def create_judged_story(
    client: OpenAI, request: str, category: str
) -> tuple[str, JudgeResult, int]:
    """Draft and revise at most twice, retaining the highest-scoring version."""
    story = generate_story(client, request, category)
    result = judge_story(client, request, category, story)
    best_story, best_result = story, result
    revisions = 0

    while not result.passed and revisions < MAX_REVISION_ROUNDS:
        revisions += 1
        story = generate_story(
            client,
            request,
            category,
            previous_story=story,
            revision_instructions="\n".join(f"- {item}" for item in result.feedback),
        )
        result = judge_story(client, request, category, story)
        if result.passed or result.total > best_result.total:
            best_story, best_result = story, result

    return best_story, best_result, revisions


def run() -> None:
    client = make_client()
    request = input("What kind of story do you want to hear? ").strip()
    if not request:
        request = EXAMPLE_REQUEST
        print(f"Using example request: {request}")

    category = route_request(client, request)
    print(f"\nRouted as: {category}")
    story, result, revisions = create_judged_story(client, request, category)
    print(f"Judge: {result.summary()} (automatic revisions: {revisions})")

    while True:
        print(f"\n{story}\n")
        change = input(
            "Want any changes? (e.g., 'make the dragon friendlier') "
            "or press Enter to finish. "
        ).strip()
        if not change:
            print("Good night!")
            return

        story = generate_story(
            client,
            request,
            category,
            previous_story=story,
            revision_instructions=f"The user requested this change: {change}",
        )
        result = judge_story(client, request, category, story)
        print(f"\nJudge: {result.summary()}")
        if result.feedback:
            print("Editor note: " + " ".join(result.feedback))


def main() -> int:
    try:
        run()
    except (RuntimeError, ValueError, OpenAIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGood night!")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
