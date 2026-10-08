"""A routed, judge-guided interactive bedtime story generator.

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

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


MODEL = "gpt-3.5-turbo"
CATEGORIES = ("adventure", "mystery", "funny/silly", "calm bedtime")
MAX_REVISION_ROUNDS = 2
EXIT_RESPONSES = {"no", "n", "done", "quit", "exit"}
SAFETY_ACKNOWLEDGMENT = (
    "I keep endings gentle for bedtime — here's a wistful one instead."
)
SAFETY_TRIGGER_PATTERNS = (
    r"\bsad\s+(?:ending|tone)\b",
    r"\bdark\s+(?:ending|tone)\b",
    r"\bscary\b",
    r"\bdies?\b",
    r"\bdeath\b",
    r"\bhorror\b",
    r"\bcreepy\b",
    r"\bfrightening\b",
    r"\bterrifying\b",
)
DEFAULT_CHOICES = (
    "Ask a friendly helper for advice.",
    "Look nearby for a clever, gentle solution.",
)
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

OPENING_CATEGORY_GUIDANCE = {
    "adventure": (
        "Launch a colorful quest with teamwork and mild, non-frightening stakes. "
        "Stop when the hero must choose how to move the quest forward."
    ),
    "mystery": (
        "Plant two fair, concrete clues and build curiosity rather than fear. "
        "Stop before the clues are explained."
    ),
    "funny/silly": (
        "Build a harmless comic problem with wordplay or repetition. Stop where "
        "two different actions could each create a satisfying comic payoff."
    ),
    "calm bedtime": (
        "Use soft sensory detail, low stakes, and unhurried pacing. Make the "
        "decision interesting without disrupting the cozy mood."
    ),
}

ENDING_CATEGORY_GUIDANCE = {
    "adventure": (
        "Let the chosen action demonstrate courage or teamwork, then bring "
        "everyone safely home."
    ),
    "mystery": (
        "Use the chosen action to connect the planted clues and reveal a gentle, "
        "satisfying explanation."
    ),
    "funny/silly": (
        "Let the chosen action pay off an earlier joke or comic pattern without "
        "making anyone the target of cruelty."
    ),
    "calm bedtime": (
        "Let the chosen action unfold softly, then gradually settle the energy "
        "into a cozy, sleepy final image."
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
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to `.env` or export it in your "
            "shell, then run `python main.py` again."
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


def needs_safety_acknowledgment(request: str) -> bool:
    """Detect requests whose tone the bedtime safety prompt will soften."""
    normalized = request.casefold()
    return any(re.search(pattern, normalized) for pattern in SAFETY_TRIGGER_PATTERNS)


def generate_opening(client: OpenAI, request: str, category: str) -> str:
    """Write only the setup and complication, ending with a hidden dilemma."""
    system_prompt = f"""You create interactive bedtime adventures for ages 5-10.

Write ONLY the setup and complication. The story body before the DECISION line
must be 200-300 words; aim for about 250 words across 4-6 natural prose
paragraphs. Begin with a specific sensory detail, quickly introduce a likable
main character and their want, then let one surprising but non-frightening
problem grow naturally from the user's details. Stop at a meaningful decision
where either of two safe actions could move the story forward. Do not solve the
problem, hint at an ending, list choices, or write events after the decision.
Do not use headings or structural labels such as "Setup" or "Complication."

Use clear, mostly simple sentences. Include no violence, cruelty, romance,
frightening imagery, or unsafe behavior. Preserve the user's names, animals,
setting, and other concrete details.

The routed style is {category}. {OPENING_CATEGORY_GUIDANCE[category]}

Finish with a separate final line in exactly this format:
DECISION: <one sentence describing what the main character must decide>

The DECISION line is metadata for another agent. Do not refer to that label in
the story. Return only the opening and the required final line."""
    return call_model(
        client,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Story request:\n{request}"},
        ],
        temperature=0.8,
        max_tokens=700,
    )


def split_opening_decision(opening: str) -> tuple[str, str]:
    """Remove DECISION metadata from an opening and return its dilemma."""
    story_lines: list[str] = []
    dilemma = "How should the main character solve the problem?"
    for line in opening.splitlines():
        match = re.match(r"^\s*DECISION:\s*(.+?)\s*$", line, flags=re.I)
        if match:
            dilemma = match.group(1)
        else:
            story_lines.append(line)
    return "\n".join(story_lines).strip(), dilemma


def generate_choices(client: OpenAI, opening: str, category: str) -> list[str]:
    """Generate exactly two safe paths, falling back if JSON is malformed."""
    prompt = f"""Create two choices for a child reading an interactive story.

The choices must respond directly to the final DECISION dilemma, be clearly
different in approach (not paraphrases), and give the hero meaningful agency.
Both must be safe, kind, plausible in the story, and capable of leading to a
gentle resolution. Use active language a 5-10 year old understands. Each choice
must be one short sentence and must not reveal what happens afterward. Match the
{category!r} tone.

Return only valid JSON in exactly this shape:
{{"choices": ["First choice.", "Second choice."]}}

Opening and dilemma:
{opening}"""
    raw = call_model(
        client,
        [
            {
                "role": "system",
                "content": (
                    "You design delightful, safe choices for children's stories."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
        json_mode=True,
    )
    try:
        choices = _parse_json_object(raw).get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 2
            or not all(
                isinstance(choice, str) and choice.strip() for choice in choices
            )
        ):
            raise ValueError("choices must contain exactly two non-empty strings")
        cleaned = [choice.strip() for choice in choices]
        if cleaned[0].casefold() == cleaned[1].casefold():
            raise ValueError("choices must be distinct")
        return cleaned
    except (json.JSONDecodeError, ValueError):
        return list(DEFAULT_CHOICES)


def infer_hero(request: str) -> str:
    """Find a supplied character name for the choice prompt when possible."""
    match = re.search(r"\bnamed\s+([A-Z][A-Za-z'-]*)", request)
    return match.group(1) if match else "the hero"


def prompt_for_choice(hero: str, choices: list[str]) -> str | None:
    """Collect one of two paths; return None when the reader wants to finish."""
    print(f"\nWhat should {hero} do?")
    print(f"  1. {choices[0]}")
    print(f"  2. {choices[1]}")

    answer = input("Pick 1 or 2 (or press Enter for 1): ").strip().casefold()
    if answer in EXIT_RESPONSES:
        return None
    if answer in ("", "1"):
        return choices[0]
    if answer == "2":
        return choices[1]

    print("Please choose 1 or 2.")
    answer = input("Pick 1 or 2 (or press Enter for 1): ").strip().casefold()
    if answer in EXIT_RESPONSES:
        return None
    return choices[1] if answer == "2" else choices[0]


def generate_ending(
    client: OpenAI,
    opening: str,
    choice: str,
    request: str,
    category: str,
    *,
    previous_ending: str | None = None,
    revision_instructions: str | None = None,
) -> str:
    """Continue the fixed opening and resolve the reader's chosen path."""
    system_prompt = f"""You complete interactive bedtime stories for ages 5-10.

Write ONLY the continuation and gentle resolution. Write 200-300 words and aim
for about 250. Continue directly from the supplied opening without recapping
it. Make the reader's chosen action cause what happens next; show a small
consequence, discovery, or act of teamwork that could not belong to the other
path. Resolve the central complication clearly, echo one concrete detail from
the opening, and finish with a warm final image that feels satisfying at
bedtime. Use natural prose without structural headings or labels.

Use clear, mostly simple sentences. Include no scary or violent content,
romance, cruelty, or unsafe behavior. Keep the ending positive and emotionally
reassuring. Do not offer another choice and do not mention prompts or agents.

The routed style is {category}. {ENDING_CATEGORY_GUIDANCE[category]}"""
    user_parts = [
        f"Original request:\n{request}",
        f"Opening (do not rewrite it):\n{opening}",
        f"Reader's chosen path (honor it visibly):\n{choice}",
    ]
    if previous_ending:
        user_parts.append(f"Ending to improve:\n{previous_ending}")
    if revision_instructions:
        user_parts.append(
            "Required editor feedback (revise only the ending):\n"
            f"{revision_instructions}"
        )
    return call_model(
        client,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        temperature=0.8,
        max_tokens=700,
    )


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
        raise ValueError("model output was not a JSON object")
    return value


def judge_story(
    client: OpenAI,
    request: str,
    category: str,
    story: str,
    chosen_path: str = "No explicit reader choice was supplied.",
) -> JudgeResult:
    """Score a draft consistently and return specific revision guidance."""
    rubric_keys = (
        "age_appropriateness",
        "story_arc",
        "engagement",
        "category_fit",
        "safety",
        "choice_honored",
    )
    prompt = f"""Evaluate the story against the request and routed category.

Score each criterion from 1 (poor) to 5 (excellent):
1. age_appropriateness: safe, understandable language and themes for ages 5-10
2. story_arc: setup, complication, and gentle resolution are all present
3. engagement: vivid, fun, and likely to hold a child's attention
4. category_fit: the story matches the {category!r} tone
5. safety: nothing scary, violent, romantic, cruel, or inappropriate
6. choice_honored: the chosen action visibly causes the resolution; do not award
   4 or 5 if the ending could follow either choice unchanged

Verdict is "pass" only when every score is at least 4; otherwise it is "fail".
For a failure, feedback must contain concise, actionable edits tied to weak
criteria. For a pass, feedback may be empty. Return only this JSON shape:
{{"verdict":"pass or fail","scores":{{"age_appropriateness":1,"story_arc":1,
"engagement":1,"category_fit":1,"safety":1,"choice_honored":1}},
"feedback":["specific edit"]}}

Original request:
{request}

Reader's chosen path:
{chosen_path}

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
    client: OpenAI,
    request: str,
    category: str,
    opening: str,
    choice: str,
) -> tuple[str, JudgeResult, int]:
    """Generate and revise only the ending, retaining the strongest full story."""
    ending = generate_ending(client, opening, choice, request, category)
    story = f"{opening}\n\n{ending}"
    result = judge_story(client, request, category, story, choice)
    best_story, best_result = story, result
    revisions = 0

    while not result.passed and revisions < MAX_REVISION_ROUNDS:
        revisions += 1
        ending = generate_ending(
            client,
            opening,
            choice,
            request,
            category,
            previous_ending=ending,
            revision_instructions="\n".join(f"- {item}" for item in result.feedback),
        )
        story = f"{opening}\n\n{ending}"
        result = judge_story(client, request, category, story, choice)
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
    if needs_safety_acknowledgment(request):
        print(SAFETY_ACKNOWLEDGMENT)

    raw_opening = generate_opening(client, request, category)
    opening, dilemma = split_opening_decision(raw_opening)
    choices = generate_choices(
        client,
        f"{opening}\n\nDECISION: {dilemma}",
        category,
    )
    print(f"\n{opening}\n")
    choice = prompt_for_choice(infer_hero(request), choices)
    if choice is None:
        print("Good night!")
        return

    story, result, revisions = create_judged_story(
        client, request, category, opening, choice
    )
    print(f"Judge: {result.summary()} (automatic revisions: {revisions})")

    # The opening was already shown before the choice; reveal only its conclusion.
    ending = story.removeprefix(opening).strip()
    print(f"\n{ending}\n")

    while True:
        change = input(
            "Want any changes? (e.g., 'make the dragon friendlier') "
            "Type 'no' or press Enter to finish. "
        ).strip()
        if not change or change.casefold() in EXIT_RESPONSES:
            print("Good night!")
            return

        story = generate_story(
            client,
            request,
            category,
            previous_story=story,
            revision_instructions=f"The user requested this change: {change}",
        )
        result = judge_story(client, request, category, story, choice)
        print(f"\nJudge: {result.summary()}")
        if result.feedback:
            print("Editor note: " + " ".join(result.feedback))
        print(f"\n{story}\n")


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
