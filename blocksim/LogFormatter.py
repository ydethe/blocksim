import logging


class LogFormatter(logging.Formatter):
    dbg_fmt = "[%(levelname)s] - %(asctime)s - L%(lineno)d@%(filename)s - %(message)s"

    def __init__(self):
        fmt = "[%(levelname)s] - %(message)s"

        super().__init__(fmt=fmt, datefmt=None, style="%")

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = LogFormatter.dbg_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result
