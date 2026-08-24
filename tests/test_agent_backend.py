from src.agent import AgentProblem, LocalVerifierBackend
from src.verifier import TestCase as JudgeTestCase


def test_local_backend_exposes_visible_failure_but_not_hidden_tests() -> None:
    problem = AgentProblem(
        problem_id="toy:add",
        problem="Add two integers.",
        visible_tests=(JudgeTestCase(input="1 2\n", output="3\n"),),
        hidden_tests=(JudgeTestCase(input="100 200\n", output="300\n"),),
    )
    code = "#include <iostream>\nint main(){std::cout << 0 << '\\n';}"
    backend = LocalVerifierBackend()
    observation = backend.execute_visible(
        code, problem, executions_remaining=2, max_feedback_bytes=4096
    )
    hidden = backend.evaluate_hidden(code, problem)
    assert observation.status == "wrong_answer"
    assert "1 2" in observation.model_feedback
    assert "Expected output:\n3" in observation.model_feedback
    assert "100 200" not in observation.model_feedback
    assert not hidden.success


def test_feedback_is_byte_bounded() -> None:
    long_input = "x" * 10000
    problem = AgentProblem(
        problem_id="toy:bounded",
        problem="Print nothing.",
        visible_tests=(JudgeTestCase(input=long_input, output="expected"),),
        hidden_tests=(JudgeTestCase(input="", output="expected"),),
    )
    observation = LocalVerifierBackend().execute_visible(
        "int main(){}", problem, executions_remaining=1, max_feedback_bytes=256
    )
    assert len(observation.model_feedback.encode("utf-8")) <= 256
    assert "[truncated" in observation.model_feedback
