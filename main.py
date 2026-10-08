"""A routed, judge-guided interactive bedtime story generator.
With two more hours, I would add golden-set evals, persistent character memory,
streaming, and parent-facing reading-level and content-safety signals."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


MODEL = "gpt-3.5-turbo"
CATEGORIES = ("adventure", "mystery", "funny/silly", "calm bedtime")
RUBRIC = ("age_appropriateness", "story_arc", "engagement", "category_fit",
          "safety", "choice_honored")
MAX_REVISION_ROUNDS = 2
EXIT_RESPONSES = {"no", "n", "done", "quit", "exit"}
SAFETY_TRIGGERS = (r"\bsad\b", r"\bdark\s+(?:ending|tone)\b",
                   r"\b(?:scary|dies?|death|horror|creepy|frightening|terrifying)\b")
SAFETY_ACKNOWLEDGMENT = "I keep endings gentle for bedtime — here's a wistful one instead."
DEFAULT_CHOICES = ("Ask a friendly helper for advice.",
                   "Look nearby for a clever, gentle solution.")
CHANGE_HINT_TEMPLATES = (
    "make the {creature} friendlier", "give {hero} a new friend",
    "add a silly surprise for {hero}", "make the ending even cozier",
)
EXAMPLE_REQUEST = "A story about a girl named Alice and her best friend Bob, who is a cat."
STYLE_GUIDANCE = {
    "adventure": "Use a colorful quest, teamwork, and mild, non-frightening stakes.",
    "mystery": "Plant fair clues, build curiosity rather than fear, and reveal them gently.",
    "funny/silly": "Use kind wordplay, comic repetition, and harmless slapstick.",
    "calm bedtime": "Use soft sensory detail, low stakes, and unhurried, cozy pacing.",
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
        scores = ", ".join(f"{name.replace('_', ' ')} {score}/5"
                           for name, score in self.scores.items())
        return f"{self.verdict.upper()} — {scores}"


def make_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in `.env` or your shell, then retry.")
    return OpenAI(api_key=key)


def call_model(client: OpenAI, messages: list[dict[str, str]], temperature: float,
               max_tokens: int, json_mode: bool = False) -> str:
    request: dict[str, Any] = dict(model=MODEL, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    content = client.chat.completions.create(**request).choices[0].message.content
    if not content:
        raise RuntimeError("The model returned an empty response.")
    return content.strip()


def route_request(client: OpenAI, request: str) -> str:
    instruction = "Reply with one label: adventure, mystery, funny/silly, or calm bedtime."
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": request}]
    category = call_model(client, messages, 0.0, 10).lower().rstrip(".")
    return next((item for item in CATEGORIES if item in category), "calm bedtime")


def generate_opening(client: OpenAI, request: str, category: str) -> str:
    prompt = f"""Write only the setup and complication of an interactive bedtime
story for ages 5-10. Write 200-300 words; do not end before 200. Use 4-6 paragraphs.
Open with a sensory detail, establish a likable hero and want, then grow one
surprising, non-frightening problem from the user's details. Stop at a meaningful
decision with two possible safe actions. Do not solve the problem, list choices,
or use headings such as Setup or Complication. Preserve supplied names, animals,
and settings. Include no violence, cruelty, romance, fear, or unsafe behavior.
Style: {category}. {STYLE_GUIDANCE[category]}
End with exactly two non-story lines in this form:
DECISION: <one sentence describing what the hero must decide>\nHERO: <the hero's name>
Do not add a metadata heading. Return only the opening and these two lines."""
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": request}]
    return call_model(client, messages, 0.8, 700)


def generate_choices(client: OpenAI, opening: str, category: str) -> list[str]:
    prompt = f"""Create exactly two choices for this interactive children's story.
They must directly answer the DECISION, use clearly different approaches, and
give the hero meaningful agency. Both must be safe, kind, plausible, leadable to
a gentle resolution, and simple enough for ages 5-10. Start each with a verb;
do not number or label them, reveal outcomes, or break the {category!r} tone.
Return only valid JSON in this shape: {{"choices": ["<first choice>", "<second choice>"]}}.
Invent both choices yourself; never output the placeholder text.
{opening}"""
    messages = [{"role": "system", "content": "Design delightful, safe choices."},
                {"role": "user", "content": prompt}]
    raw = call_model(client, messages, 0.7, 100, True)
    try:
        choices = json.loads(raw).get("choices")
        if not isinstance(choices, list) or len(choices) != 2:
            raise ValueError
        choices = [choice.strip() for choice in choices]
        if not all(choices) or not all(isinstance(choice, str) for choice in choices):
            raise ValueError
        if choices[0].casefold() == choices[1].casefold():
            raise ValueError
        return choices
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return list(DEFAULT_CHOICES)


def prompt_for_choice(hero: str, choices: list[str]) -> str | None:
    print(f"\nWhat should {hero} do?\n  1. {choices[0]}\n  2. {choices[1]}")
    for attempt in range(2):
        answer = input("Pick 1 or 2 (or press Enter for 1): ").strip().casefold()
        if answer in EXIT_RESPONSES:
            return None
        if answer in ("", "1"):
            return choices[0]
        if answer == "2":
            return choices[1]
        if attempt == 0:
            print("Please choose 1 or 2.")
    return choices[0]


def build_change_hint(hero: str, opening: str) -> str:
    pattern = r"\b(dragon|cat|dog|rabbit|fox|owl|unicorn|bear|wolf|mouse|bird|dinosaur)\b"
    creature = re.search(pattern, opening, re.I)
    templates = [item for item in CHANGE_HINT_TEMPLATES
                 if creature or "{creature}" not in item]
    return random.choice(templates).format(hero=hero,
                                            creature=creature.group(1) if creature else "")


def generate_ending(
    client: OpenAI,
    opening: str,
    choice: str,
    request: str,
    category: str,
    previous_ending: str | None = None,
    revision_instructions: str | None = None,
) -> str:
    prompt = f"""Write only the 200-300 word continuation and gentle resolution
of this bedtime story for ages 5-10. Continue without recapping. The chosen action
must visibly cause a consequence, discovery, or act of teamwork unique to its
path. Resolve the central problem, echo one opening detail, and end on a warm
image. Do not introduce new characters, treasures, or subplots — the resolution
must be caused by the chosen action, and the final image must reference it.
Use natural, simple prose with no headings, new choice, violence, fear,
romance, cruelty, unsafe behavior, or prompt discussion.
Style: {category}. {STYLE_GUIDANCE[category]}"""
    context = [
        f"Original request:\n{request}",
        f"Fixed opening:\n{opening}",
        f"Chosen path:\n{choice}",
    ]
    if previous_ending:
        context.append(f"Ending to improve:\n{previous_ending}")
    if revision_instructions:
        context.append(f"Revise only the ending using this feedback:\n{revision_instructions}")
    return call_model(
        client,
        [{"role": "system", "content": prompt}, {"role": "user", "content": "\n\n".join(context)}],
        0.8,
        700,
    )


def generate_story(
    client: OpenAI,
    request: str,
    category: str,
    previous_story: str,
    revision_instructions: str,
) -> str:
    prompt = f"""Rewrite this complete 400-600 word bedtime story for ages 5-10.
Apply the user's change while preserving supplied details and the setup,
complication, and gentle resolution. Use clear prose with no violence, fear,
romance, cruelty, or unsafe behavior. End positively.
Style: {category}. {STYLE_GUIDANCE[category]}

Original request:\n{request}\n\nStory:\n{previous_story}

Requested change:\n{revision_instructions}"""
    return call_model(client, [{"role": "system", "content": prompt}], 0.8, 1_200)


def judge_story(
    client: OpenAI,
    request: str,
    category: str,
    story: str,
    chosen_path: str = "No explicit reader choice was supplied.",
) -> JudgeResult:
    prompt = f"""Score this children's story from 1 (poor) to 5 (excellent) on:
- age_appropriateness: safe, understandable language and themes for ages 5-10
- story_arc: setup, complication, and gentle resolution are all present
- engagement: vivid, fun, and likely to hold a child's attention
- category_fit: matches the {category!r} tone
- safety: nothing scary, violent, romantic, cruel, or inappropriate
- choice_honored: the chosen action visibly causes the resolution; score below 4
  only if swapping in the other choice would leave the problem's solution unchanged
Verdict is pass only if every score is at least 4. For a failure, give concise,
actionable feedback tied to weak criteria. Return only valid JSON:
{{"verdict":"pass or fail","scores":{{"age_appropriateness":1,"story_arc":1,
"engagement":1,"category_fit":1,"safety":1,"choice_honored":1}},
"feedback":["specific edit"]}}
Request: {request}\nChosen path: {chosen_path}\n\nStory:\n{story}"""
    data = json.loads(
        call_model(
            client,
            [
                {"role": "system", "content": "Be a strict children's-story editor."},
                {"role": "user", "content": prompt},
            ],
            0.1,
            500,
            True,
        )
    )
    raw_scores = data.get("scores", {})
    scores = {name: raw_scores.get(name) for name in RUBRIC}
    if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
        raise ValueError("Judge returned invalid scores.")
    feedback = data.get("feedback") or []
    if not isinstance(feedback, list) or not all(isinstance(item, str) for item in feedback):
        raise ValueError("Judge returned invalid feedback.")
    verdict = "pass" if all(score >= 4 for score in scores.values()) else "fail"
    if verdict == "fail" and not feedback:
        feedback = ["Strengthen every criterion scoring below 4."]
    return JudgeResult(verdict, scores, feedback)


def create_judged_story(
    client: OpenAI, request: str, category: str, opening: str, choice: str
) -> tuple[str, JudgeResult, int]:
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
            ending,
            "\n".join(f"- {item}" for item in result.feedback),
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
    if any(re.search(pattern, request.casefold()) for pattern in SAFETY_TRIGGERS):
        print(SAFETY_ACKNOWLEDGMENT)
    raw_opening = generate_opening(client, request, category)
    match = re.search(r"(?im)^\s*DECISION:\s*(.+?)\s*$", raw_opening)
    dilemma = match.group(1) if match else "How should the hero solve the problem?"
    hero_match = re.search(r"(?im)^\s*HERO:\s*(.+?)\s*$", raw_opening)
    hero = hero_match.group(1) if hero_match else "the hero"
    opening = re.sub(r"(?im)^\s*(?:metadata:|(?:DECISION|HERO):\s*.+?)\s*$", "", raw_opening).strip()
    choices = generate_choices(client, f"{opening}\n\nDECISION: {dilemma}", category)
    print(f"\n{opening}\n")
    choice = prompt_for_choice(hero, choices)
    if choice is None:
        print("Good night!")
        return

    story, result, revisions = create_judged_story(
        client, request, category, opening, choice
    )
    print(f"Judge: {result.summary()} (automatic revisions: {revisions})")
    print(f"\n{story.removeprefix(opening).strip()}\n")
    while True:
        hint = build_change_hint(hero, opening)
        change = input(
            f"Want any changes? (e.g., '{hint}') "
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
            revision_instructions=change,
        )
        result = judge_story(client, request, category, story, choice)
        print(f"\nJudge: {result.summary()}")
        if result.feedback:
            print("Editor note: " + " ".join(result.feedback))
        print(f"\n{story}\n")


def main() -> int:
    try:
        run()
        return 0
    except (RuntimeError, ValueError, OpenAIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGood night!")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
