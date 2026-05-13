import pytest

from neural_net.optimizer import SgdOptimizer


@pytest.fixture
def sgd_optimizer() -> SgdOptimizer:
    return SgdOptimizer([])


def test_sgd_optimizer_registry(sgd_optimizer: SgdOptimizer):
    assert "sgd" in SgdOptimizer.REGISTRY


def test_sgd_optimizer_zero_grad(sgd_optimizer: SgdOptimizer):
    assert sgd_optimizer.zero_grad() is None


def test_sgd_optimizer_step(sgd_optimizer: SgdOptimizer):
    assert sgd_optimizer.step() is None
