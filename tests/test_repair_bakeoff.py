from scripts.prepare_repair_bakeoff import select


def test_bakeoff_selection_is_deterministic_and_balanced() -> None:
    rows = [
        {"task_id": f"{difficulty}-{index}", "problem": {"difficulty": difficulty}}
        for difficulty in ("easy", "medium", "hard", "unknown")
        for index in range(4)
    ]
    first = select(rows, size=10, seed=7)
    second = select(list(reversed(rows)), size=10, seed=7)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]
    assert {row["problem"]["difficulty"] for row in first} == {
        "easy",
        "medium",
        "hard",
        "unknown",
    }
