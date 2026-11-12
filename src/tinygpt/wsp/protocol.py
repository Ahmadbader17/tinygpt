"""The data protocol of the study: which stories and prompts are dev, and which are sealed.

Dev data is the repo's existing validation subset (the first 5,000 stories of the official
validation file). The sealed test data is a later, disjoint range of the same file. This
module only *defines* the sealed range; the one place allowed to download and read it is the
final evaluation script. Rules and reasons: experiments/wsp/PLAN.md section 3.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tinygpt.data.prepare import read_clean_split

# --- dev (the editor may use these) ---------------------------------------------------------
DEV_STORY_FILE = Path("data/raw/val.txt")  # written by `make data`: valid-file stories 0..4999
DEV_LOSS_STORIES = range(0, 2500)  # dev stories used for loss and activation analysis
DEV_PROMPT_STORIES = range(2500, 5000)  # dev stories whose openings become dev prompts
N_DEV_PROMPTS = 300

# --- sealed test (final evaluation only) ----------------------------------------------------
# Pinned to a dataset commit so the bytes cannot change under us. sha256 from the Hub's
# file metadata, recorded on 2026-11-11 without downloading the file.
SEALED_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SEALED_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{SEALED_REVISION}/TinyStoriesV2-GPT4-valid.txt"
)
SEALED_SHA256 = "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
SEALED_LOSS_STORIES = range(5000, 7000)  # raw story indices in the validation file
SEALED_PROMPT_STORIES_FROM = 7000  # openings of stories from here on become test prompts
N_SEALED_PROMPTS = 300

# --- prompts ---------------------------------------------------------------------------------
PROMPT_WORDS = 8  # a prompt is the first 8 words of a story, like the README's short openings
SAMPLING_SEEDS = (0, 1, 2, 3, 4)
MAX_NEW_TOKENS = 200

README_PROMPTS = (
    "Once upon a time",
    "One day, a little girl named",
    "The dog was very",
    "Tom and his mom went to the",
    "The sun was shining and",
    "The little robot wanted to",
    "Mia found a shiny key under her bed.",
    "It was raining, so the two friends",
    "A tiny dragon lived in",
    "Dad said, \"Let's bake a",
)


def prompt_from_story(story: str, n_words: int = PROMPT_WORDS) -> str | None:
    """The first `n_words` words of the story's first line, or None if the line is shorter."""
    words = story.split("\n", 1)[0].split()
    if len(words) < n_words:
        return None
    return " ".join(words[:n_words])


def select_prompts(
    stories: Iterable[str], n: int, exclude: Iterable[str] = (), n_words: int = PROMPT_WORDS
) -> list[str]:
    """The first `n` distinct prompts in story order, skipping any in `exclude`."""
    seen = set(exclude)
    prompts: list[str] = []
    for story in stories:
        prompt = prompt_from_story(story, n_words)
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) == n:
            break
    return prompts


def load_dev_stories(path: Path = DEV_STORY_FILE) -> list[str]:
    """All dev stories, in file order."""
    return list(read_clean_split(path))


def dev_loss_stories(stories: list[str]) -> list[str]:
    return [stories[i] for i in DEV_LOSS_STORIES if i < len(stories)]


def dev_prompts(stories: list[str]) -> list[str]:
    """Dev prompts: openings of the second half of the dev stories, minus the README prompts."""
    pool = (stories[i] for i in DEV_PROMPT_STORIES if i < len(stories))
    return select_prompts(pool, N_DEV_PROMPTS, exclude=README_PROMPTS)
