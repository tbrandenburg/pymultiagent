"""
Main module for PyMultiAgent.

This module allows running the package with 'python -m pymultiagent'.
"""
import sys

from .main import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())