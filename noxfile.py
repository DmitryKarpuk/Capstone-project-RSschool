import nox
from nox import Session
import os

TEMP_FILE = "temp.txt"


@nox.session
def black(session: Session) -> None:
    """Run black code formatter."""
    session.install("black")
    session.run("black", "src", "tests")


@nox.session
def mypy(session: Session) -> None:
    """Type check using mypy."""
    session.install("mypy")
    session.run("mypy", "src")


@nox.session
def lint(session: Session) -> None:
    """Run flake8 code linter."""
    session.install("flake8")
    session.run("flake8", "src", "tests")


@nox.session
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        f"--output=temp.txt",
        external=True,
    )
    session.install("-r", "temp.txt")
    os.unlink("temp.txt")
    session.run("pytest")
