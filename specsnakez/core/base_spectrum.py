from __future__ import annotations
import json
import logging
import numpy as np
import pandas as pd
from scipy.integrate import simps as sp
from scipy.integrate import trapz as tp
from typing import Dict, Any, Tuple

from specsnakez.core.exceptions import InvalidSpectrumError, NoTimestampOfMeasurementSetError, SpectrumIntegrationError

logger = logging.getLogger(__name__)


class MetaSpectrum(type):
    """Metaclass responsible for the correct runtime behaviour of spectrum objects derived from it."""

    def __call__(cls, *args, **kwargs):
        """This methods ensures that no spectrum may be instantiated without the required properties that are
        essential for plotting."""
        temp = super().__call__(*args, **kwargs)

        for attr in ("x", "y", "x_unit", "y_unit", "name"):
            if not hasattr(temp, attr):
                raise InvalidSpectrumError(f'Tried to instantiate spectrum object without suitable attribute {attr}!')

        return temp


class BaseSpectrum(metaclass=MetaSpectrum):
    """This class represents a basic spectrum, having an independent array of x and a dependent array of values typically
    representing experimental data resulting from instrumental measurements. It can be used as a generic template
    for many different concrete data of this structure."""

    def __init__(self, name=None, x=None, y=None, x_unit=None, y_unit=None, timestamp=None):
        self.name = name
        self.x = x
        self.y = y
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.timestamp = timestamp

    def __lt__(self, other) -> bool:
        """Allows the comparison between two spectra based on their time of measurement."""
        if self.timestamp is None:
            raise NoTimestampOfMeasurementSetError(f'{self.name} has no timestamp of measurement.')
        else:
            return True if self.timestamp < other.timestamp else False

    def __repr__(self):
        return f'{type(self).__name__} Object with name "{self.name}"'

    # working on the data
    def yield_spectral_range(self) -> Tuple[float, float, float]:
        """returns a list containing maximum and minimum wavenumer and the number of data points"""
        return [min(self.x), max(self.x), len(self.x)]

    def get_xrange_indices(self, lower, upper) -> Tuple[int, int]:
        """Takes a high (upper) and a low (lower) target x value as argument. Returns
        the indices of the wavenumber array of the spectrum that are the borders of this interval."""
        lower_index = np.argmax(self.x >= lower)
        upper_index = np.argmax(self.x >= upper)
        return int(lower_index), int(upper_index)

    def get_xrange(self) -> np.array:
        # todo: ensure this functions work as well for y_values
        """Returns the slice of the x values in the borders of lower to upper"""
        lower, upper = self.get_xrange_indices()
        return self.x[lower, upper + 1]

    def get_nearest_index(self, x_value: float) -> int:
        """Returns the index of the spectrum's x value closest to the given x_value"""
        return int(np.argmax(self.x >= x_value))

    def normalize(self, external=None) -> np.array:
        """Normalize the spectrum's y data either to the maximum of the y values or an
        external factor"""
        return self.y / np.max(self.y) if external is None else self.y / external

    def integrate_slice(self, x_array: np.array, y_array: np.array) -> np.array:
        """Integrates the y_array which has a spacing given by the x_array. First it tries to apply
        simpson rule integration rule, but if this fails the function invokes integration via
        trapeziodal rule"""
        try:
            area = sp(y_array, x_array)
            if np.isnan(area):
                logger.warning(f"""Integration failed in spectrum {self.name} using Simpson's rule. 
                Falling back to trapezoidal rule.""")
                area = tp(y_array, x_array)
            return area
        except:
            raise SpectrumIntegrationError(f'Integration not possible for {self.name}')

    # export functions
    def properties_to_dict(self) -> Dict[str, Any]:
        temp = {
            "name": self.name,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp
        }
        return temp

    def to_pandas_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.properties_to_dict())

    def to_csv(self) -> None:
        self.to_pandas_dataframe().to_csv(self.name + ".csv", index=False, sep=";")

    def to_json(self) -> str:
        temp = self.properties_to_dict()
        return json.dumps(temp)
