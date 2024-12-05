""" Python equivalent of the tee console command. """
import sys
from typing import TextIO

class Tee:
    """
    Python equivalent of the tee command.
    Allows print / stdout text to be written to a file and echoed to stdout

    In use:

    from contextlib import redirect_stdout
    
    with redirect_stdout(Tee(open("./process.log", "w", encoding="utf8"))):
        print("hello world")
    """
    def __init__(self, output1: TextIO, output2: TextIO = sys.stdout):
        """
        Initialize a new instance.
        """
        self.output1 = output1
        self.output2 = output2

    def write(self, s):
        """
        Write the passed text and, if present, echo to the second output.
        """
        if self.output2 is not None:
            self.output2.write(s)
        self.output1.write(s)

    def flush(self):
        """
        Flush the output and, if present, the second output.
        """
        if self.output2 is not None:
            self.output2.flush()
        self.output1.flush()
