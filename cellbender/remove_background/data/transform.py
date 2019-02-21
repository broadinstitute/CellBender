"""Transformations for count data that leave dtype unchanged."""

import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from typing import Union


class DataTransform(ABC):
    """Abstract class for transformation of count data matrix/array."""

    def __init__(self):
        self.name = None
        self.scale_factor = 1.
        pass

    @abstractmethod
    def transform(self, data: Union[int, float,
                                    np.ndarray,
                                    sp.csr_matrix,
                                    sp.csc_matrix,
                                    sp.coo_matrix]) -> Union[int, float,
                                                             np.ndarray,
                                                             sp.csr_matrix,
                                                             sp.csc_matrix,
                                                             sp.coo_matrix]:
        """Define the forward transformation to be performed on input data."""
        pass

    @abstractmethod
    def inverse_transform(self, data: Union[int, float,
                                            np.ndarray,
                                            sp.csr_matrix,
                                            sp.csc_matrix,
                                            sp.coo_matrix]) -> Union[int, float,
                                                                     np.ndarray,
                                                                     sp.csr_matrix,
                                                                     sp.csc_matrix,
                                                                     sp.coo_matrix]:
        """Define the reverse transformation to be performed on output data."""
        pass


class LogTransform(DataTransform):
    """Transform input data according to

    y = \ln x

    """

    def __init__(self, scale_factor: float = 1.):
        super(LogTransform, self).__init__()
        self.name = 'log'
        self.scale_factor = scale_factor

    def transform(self, data):
        """Natural logarithm"""

        data = np.log1p(data)  # log(x + 1)
        data = data * self.scale_factor

        return data

    def inverse_transform(self, data):
        """Exponential"""

        data = data / self.scale_factor  # In-place can cause casting issues.
        data = np.expm1(data)  # exp(x) - 1

        return data


class SqrtTransform(DataTransform):
    """Transform input data according to

    y = \sqrt x

    """

    def __init__(self, scale_factor: float = 1.):
        super(SqrtTransform, self).__init__()
        self.name = 'sqrt'
        self.scale_factor = scale_factor

    def transform(self, data):
        """Square root"""

        data = np.sqrt(data)  # sqrt(x)
        data = data * self.scale_factor

        return data

    def inverse_transform(self, data):
        """Exponential"""

        data = data / self.scale_factor
        if isinstance(data, np.ndarray):
            data = np.power(data, 2)  # x**2
        else:
            data = data.power(2)  # for sparse things

        return data


class IdentityTransform(DataTransform):
    """No transformation"""

    def __init__(self):
        super(IdentityTransform, self).__init__()
        self.name = 'identity'
        self.scale_factor = 1

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
