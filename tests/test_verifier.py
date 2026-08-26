from src.verifier.judge import TestCase as JudgeTestCase, judge


SUM_CODE = r"""
#include <iostream>
int main() {
    long long a, b;
    std::cin >> a >> b;
    std::cout << a + b << '\n';
}
"""


def test_correct_code_and_multiple_test_cases() -> None:
    result = judge(
        SUM_CODE,
        [
            {"input": "1 2\n", "output": "3\n"},
            {"input": "-5 8\n", "output": "3\n"},
            {"input": "100 200\n", "output": "300\n"},
        ],
    )
    assert result.compiled
    assert result.passed == 3
    assert result.total == 3
    assert result.pass_rate == 1.0
    assert result.error_type is None
    assert result.to_dict()["compiled"] is True


def test_compile_error() -> None:
    result = judge("int main( {", [{"input": "", "output": ""}])
    assert not result.compiled
    assert result.error_type == "compile_error"
    assert result.compile_stderr


def test_wrong_answer() -> None:
    result = judge(
        "#include <iostream>\nint main() { std::cout << 7; }",
        [{"input": "", "output": "8"}],
    )
    assert result.compiled
    assert result.error_type == "wrong_answer"
    assert result.passed == 0


def test_output_comparison_matches_livecodebench_line_and_decimal_rules() -> None:
    same_numbers = judge("#include <iostream>\nint main(){std::cout << \"1.0 2.00\\n\";}", [{"input": "", "output": "1 2\n"}])
    wrong_lines = judge("#include <iostream>\nint main(){std::cout << \"1 2\\n\";}", [{"input": "", "output": "1\n2\n"}])
    assert same_numbers.passed == 1
    assert wrong_lines.error_type == "wrong_answer"


def test_runtime_error() -> None:
    code = "int main() { int* value = nullptr; *value = 1; }"
    result = judge(code, [{"input": "", "output": ""}])
    assert result.compiled
    assert result.runtime_error
    assert result.error_type == "runtime_error"


def test_infinite_loop_times_out() -> None:
    result = judge(
        "int main() { while (true) {} }",
        [{"input": "", "output": ""}],
        execution_timeout_seconds=0.2,
    )
    assert result.compiled
    assert result.timeout
    assert result.error_type == "timeout"


def test_output_limit() -> None:
    result = judge(
        "#include <iostream>\nint main() { while (true) std::cout << 'x'; }",
        [{"input": "", "output": ""}],
        execution_timeout_seconds=1.0,
        output_limit_bytes=4096,
    )
    assert result.compiled
    assert result.error_type == "output_limit"


def test_custom_output_checker_accepts_semantically_valid_alternative() -> None:
    checker = r'''
#include <fstream>
#include <string>
int main(int argc, char** argv) {
    std::ifstream input(argv[1]), output(argv[2]);
    int expected_sum, a, b;
    input >> expected_sum;
    if (!(output >> a >> b)) return 1;
    return a + b == expected_sum ? 0 : 1;
}
'''
    result = judge(
        "#include <iostream>\nint main(){int n;std::cin>>n;std::cout<<2<<' '<<n-2;}",
        [JudgeTestCase(input="10\n", output="1 9\n")],
        output_checker_source=checker,
    )
    assert result.pass_rate == 1.0
