from scripts.prepare_taco_native_sft import candidate_digest, is_unsupported, problem_id


def test_native_problem_identity_is_stable() -> None:
    assert problem_id("rev", 1, "Question") == problem_id("rev", 1, "Question")
    assert problem_id("rev", 1, "Question") != problem_id("rev", 2, "Question")
    assert candidate_digest(7, 1, "Question") == candidate_digest(7, 1, "Question")


def test_interactive_problems_are_rejected() -> None:
    assert is_unsupported({"question": "This is an interactive problem"})
    assert is_unsupported({"question": "Normal", "raw_tags": "['interactive']"})
    assert not is_unsupported({"question": "Read two integers"})
