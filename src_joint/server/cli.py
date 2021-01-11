"""
LAL-Parser-Server's command line interface.
"""

import logbook
import click_completion

from . import cli_main

import serve # change to .serve or lal-parser-server.serve

if __name__ == "__main__":
    click_completion.init()
    if __debug__:
        logbook.StderrHandler(logbook.DEBUG).push_application()
    else:
        logbook.StderrHandler(logbook.INFO).push_application()
    logbook.compat.redirect_logging()
    cli_main(prog_name="lal-parser-server")
