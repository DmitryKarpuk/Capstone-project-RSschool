from click.testing import CliRunner
import pytest
import os


from forest_ml.train import train

parent_path = os.getcwd()
default_arg = [
    "-d",
    parent_path + r"\tests\test_data\pytest_train.csv",
    "-p",
    parent_path + r"\tests\test_data\estimate_params.json",
    "-st",
    "kfold_cv",
]


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_cv(runner: CliRunner) -> None:
    """Fails when CV parameters less then 1"""
    with runner.isolated_filesystem():
        path = os.getcwd()
        arglist = default_arg + ["-s", path + "/model.joblib", "--cv", "0"]
        result = runner.invoke(train, arglist)
    assert result.exit_code == 1, "CV should be more the 2"


def test_cv(runner: CliRunner) -> None:
    """OK when CV parameters more then 1"""
    with runner.isolated_filesystem():
        path = os.getcwd()
        arglist = default_arg + ["-s", path + "/model.joblib", "--cv", "2"]
        result = runner.invoke(train, arglist)
    assert result.exit_code == 0, "Invalid cv parameter"


def test_default_option(runner: CliRunner) -> None:
    """OK if train works without error on default option"""
    with runner.isolated_filesystem():
        path = os.getcwd()
        arglist = default_arg + ["-s", path + "/model.joblib"]
        result = runner.invoke(train, arglist)
    assert result.exit_code == 0, "Train fuction is faild"
