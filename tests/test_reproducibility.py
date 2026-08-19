import random

from src.utils.reproducibility import set_seed


def test_python_random_is_reproducible() -> None:
    set_seed(123)
    first = [random.random() for _ in range(3)]
    set_seed(123)
    assert [random.random() for _ in range(3)] == first

