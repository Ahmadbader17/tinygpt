from tinygpt.wsp import protocol
from tinygpt.wsp.protocol import prompt_from_story, select_prompts


def test_prompt_is_the_first_words_of_the_first_line() -> None:
    story = "Once upon a time, there was a little girl named Lily.\nShe was happy."
    assert prompt_from_story(story) == "Once upon a time, there was a little"
    assert prompt_from_story("Too short.\nSecond line has many more words than eight.") is None


def test_select_prompts_skips_duplicates_and_excluded() -> None:
    stories = [
        "a b c d e f g h one",
        "a b c d e f g h two",  # same first 8 words: duplicate prompt
        "x b c d e f g h",
        "short",
        "y b c d e f g h",
        "z b c d e f g h",
    ]
    got = select_prompts(stories, 2, exclude=["x b c d e f g h"])
    assert got == ["a b c d e f g h", "y b c d e f g h"]


def test_sealed_stories_are_disjoint_from_dev_stories() -> None:
    # `make data` keeps validation stories 0..4999; the sealed ranges must start after them
    dev_end = max(protocol.DEV_LOSS_STORIES.stop, protocol.DEV_PROMPT_STORIES.stop)
    assert dev_end <= 5000 <= protocol.SEALED_LOSS_STORIES.start
    assert protocol.SEALED_LOSS_STORIES.stop <= protocol.SEALED_PROMPT_STORIES_FROM
    assert set(protocol.DEV_LOSS_STORIES).isdisjoint(protocol.DEV_PROMPT_STORIES)
    assert protocol.SEALED_REVISION in protocol.SEALED_URL
