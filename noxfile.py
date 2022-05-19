import nox
from nox.sessions import Session

nox.options.sessions = ["black", "lint", "mypy", "test"]
location = ["src/", "tests/"]


@nox.session
def lint(session):
    session.install("flake8")
    session.run("flake8", "src/")


@nox.session
def black(session):
    session.install("black")
    session.run("black", ".")


@nox.session
def mypy(session):
    session.install("mypy")
    session.run("mypy", "src/")


@nox.session
def test(session):
    """Run the test suite."""
    session.install("poetry")
    session.install("pytest")
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", external=True)
