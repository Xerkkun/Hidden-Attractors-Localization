from __future__ import annotations

import subprocess
import sys
import pytest

@pytest.mark.cli
def test_cli_help_commands():
    # List of subcommands to test help
    commands = [
        [],
        ["inspect", "--help"],
        ["validate", "--help"],
        ["seed", "--help"],
        ["continuation", "--help"],
        ["bifurcation", "--help"],
        ["lyapunov", "--help"],
        ["chaos-test", "--help"],
    ]

    for cmd in commands:
        full_args = [sys.executable, "-m", "hidden_attractors.cli.main"] + cmd
        res = subprocess.run(
            full_args,
            capture_output=True,
            text=True,
            check=False
        )
        assert res.returncode == 0 or (cmd == [] and res.returncode != 0), \
            f"Command {full_args} failed with returncode {res.returncode}. stderr: {res.stderr}"
        
        # If we passed --help, exit code is 0 and output should mention usage
        if "--help" in cmd:
            assert "usage: hidden-attractors" in res.stdout, f"Expected usage help in stdout for {cmd}. Got: {res.stdout}"
            assert res.returncode == 0
        else:
            # Without arguments, the parser should fail because 'group' is required, showing usage
            assert "usage: hidden-attractors" in res.stderr or "usage: hidden-attractors" in res.stdout
