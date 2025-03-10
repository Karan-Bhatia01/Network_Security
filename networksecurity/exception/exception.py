import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        if exc_tb:  # Ensuring traceback exists to avoid NoneType errors
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = "Unknown"

    def __str__(self):
        return "Error occurred in Python script: [{0}], Line number: [{1}], Error message: [{2}]".format(
            self.file_name, self.lineno, self.error_message
        )
