from scripts.audit_codecontests_plus import cpp_submissions, prepare_contest_submission, stable_sample
from scripts.prepare_codecontests_plus_repair import excluded_problem_ids


def test_cpp_submissions_filters_language_and_empty_code() -> None:
    rows = [
        {"language": "C++", "code": "int main() {}"},
        {"language": "Python3", "code": "print(1)"},
        {"language": "C++", "code": ""},
    ]
    assert cpp_submissions(rows) == ["int main() {}"]


def test_stable_sample_is_deterministic() -> None:
    rows = [
        {"source": "Codeforces", "id": str(index)}
        for index in range(10)
    ]
    assert stable_sample(rows, 4, 7) == stable_sample(list(reversed(rows)), 4, 7)


def test_prepare_contest_submission_defines_online_judge() -> None:
    prepared = prepare_contest_submission("int main() {}")
    assert "#define ONLINE_JUDGE 1" in prepared
    assert prepared.endswith("int main() {}")


def test_excluded_problem_ids_loads_compact_indices(tmp_path) -> None:
    index = tmp_path / "problems.index.json"
    index.write_text(
        '{"schema_version":"jsonl-byte-offset-index-v1","records":'
        '{"ccplus:a:1":[0,10],"ccplus:b:2":[10,20]}}',
        encoding="utf-8",
    )
    assert excluded_problem_ids({"exclude_problem_indices": [str(index)]}) == {
        "ccplus:a:1",
        "ccplus:b:2",
    }


def test_excluded_problem_ids_rejects_invalid_index(tmp_path) -> None:
    index = tmp_path / "bad.index.json"
    index.write_text('{"problem_ids": []}', encoding="utf-8")
    try:
        excluded_problem_ids({"exclude_problem_indices": [str(index)]})
    except ValueError as error:
        assert "expected a records object" in str(error)
    else:
        raise AssertionError("invalid compact index should fail")
