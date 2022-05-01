from click.testing import CliRunner
import pytest

from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_cv(runner: CliRunner) -> None:
    """ Fails when CV parameters less then 1 """
    result = runner.invoke(train, ["--cv", 0])
    assert result.exit_code == 2
    assert "Invalid value for '--cv'" in result.output


def test_cv(runner: CliRunner) -> None:
    """ OK when CV parameters less then 1 """
    result = runner.invoke(train, ["--cv", 2])
    assert result.exit_code == 0
    assert "Valid value for '--cv'" in result.output
