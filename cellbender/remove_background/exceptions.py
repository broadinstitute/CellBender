"""Exceptions defined by CellBender"""


class NanException(ArithmeticError):
    """Exception raised when a NaN is present.

    Attributes:
        param: Name of parameter(s) causing / containing the NaN
    """

    def __init__(self, param: str):
        self.param = param
        self.message = 'A wild NaN appeared!  In param {' + self.param + '}'


class ElboException(ValueError):
    """Exception raised when training procedure is failing to meet expectations.

    Attributes:
        message: Message to write to log
    """

    def __init__(self, message: str):
        self.message = message
