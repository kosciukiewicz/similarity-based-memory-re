from pytest import ExitCode


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
