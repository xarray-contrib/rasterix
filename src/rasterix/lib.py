"""Shared library utilities for rasterix."""

import logging

# Define TRACE level (lower than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class TraceLogger(logging.Logger):
    """Logger with trace level support."""

    def trace(self, message, *args, **kwargs):
        """Log a message with severity 'TRACE'."""
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)


# Set the custom logger class
logging.setLoggerClass(TraceLogger)

# Create logger for the rasterix package
logger = logging.getLogger("rasterix")
