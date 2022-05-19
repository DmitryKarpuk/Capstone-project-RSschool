import os
from typing import Any
import nox
from nox.sessions import Session


nox.options.sessions = "black", "lint", "mypy", "tests"
LOCATIONS = "src", "tests", "noxfile.py"
TEMP_FILE = "temp.txt"


def install_with_constraints(
    session: Session, *args: str, **kwargs: Any
) -> None:
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        f"--output={TEMP_FILE}",
        external=True,
    )
    session.install(f"--constraint={TEMP_FILE}", *args, **kwargs)
    os.unlink(TEMP_FILE)


@nox.session
def lint(session: Session) -> None:
    """Run flake8 code linter."""
    args = session.posargs or LOCATIONS
    install_with_constraints(
        session,
        "flake8",
        "flake8-black",
        "flake8-annotations",
    )
    session.run("flake8", *args)


@nox.session
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session
def mypy(session: Session) -> None:
    """Type check using mypy."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", *args)
