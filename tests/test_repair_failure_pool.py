from scripts.build_repair_failure_pool import build_pool


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
    rows = build_pool([problem], [failed, failed], student_model="student", excluded=set())
    assert len(rows) == 1
    assert rows[0]["initial_submission"]["student_model"] == "student"
    assert build_pool([problem], [failed], student_model="student", excluded={"train:1"}) == []
    passed = {**failed, "judge": {"passed": 1, "total": 1}}
    assert build_pool([problem], [passed], student_model="student", excluded=set()) == []
