# Exceptions defined by CellBender


class NanException(ArithmeticError):
    """Exception raised when a NaN is present.

    Attributes:
        param: Name of parameter(s) causing / containing the NaN
    """

    def __init__(self, param: str):
        self.param = param
        self.message = 'A wild NaN appeared!  In param {' + self.param + '}'
