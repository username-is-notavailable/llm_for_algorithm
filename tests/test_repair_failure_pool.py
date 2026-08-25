from scripts.build_repair_failure_pool import build_one_shot_pool, build_pool


def test_failure_pool_filters_success_exclusions_and_duplicates() -> None:
    problem = {
        "problem_id": "train:1",
        "problem": "Add.",
        "language": "cpp",
        "visible_tests": [{"input": "1 2", "output": "3"}],
        "hidden_tests": [{"input": "2 3", "output": "5"}],
    }
    failed = {
        "problem_id": "train:1",
        "sample_index": 0,
        "response": "bad",
        "code": "int main(){}",
        "judge": {"passed": 0, "total": 1},
    }
    rows = build_pool([problem], [failed, failed], producer_model="teacher", excluded=set())
    assert len(rows) == 1
    assert rows[0]["initial_submission"]["producer_model"] == "teacher"
    assert build_pool([problem], [failed], producer_model="teacher", excluded={"train:1"}) == []
    passed = {**failed, "judge": {"passed": 1, "total": 1}}
    assert build_pool([problem], [passed], producer_model="teacher", excluded=set()) == []


def test_one_shot_pool_requires_clean_stopped_verified_generation() -> None:
    problem = {"problem_id": "train:1", "problem": "Add.", "difficulty": "easy"}
    generation = {
        "problem_id": "train:1",
        "response": "```cpp\nint main(){}\n```",
        "code": "int main(){}",
        "judge": {"passed": 2, "total": 2},
        "finish_reason": "stop",
        "response_tokens": 20,
    }
    rows = build_one_shot_pool(
        [problem], [generation], teacher_model="teacher", excluded=set(), max_response_tokens=100
    )
    assert len(rows) == 1 and rows[0]["teacher_model"] == "teacher"
    assert build_one_shot_pool(
        [problem], [{**generation, "finish_reason": "length"}],
        teacher_model="teacher", excluded=set(), max_response_tokens=100,
    ) == []
