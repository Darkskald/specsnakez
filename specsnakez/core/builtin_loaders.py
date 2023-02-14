from functools import partial
from typing import Callable

import pandas as pd

from specsnakez.core.base_spectrum import BaseSpectrum
from specsnakez.core.langmuir_isotherm import LtIsotherm
from specsnakez.core.sfg_spectrum import SfgSpectrum


class ExtractorFactory:
    """As in most cases spectral data appears in the shape of csv-like files with a varying amount of columns,
    separators etc., this class has the purpose to create custom extractor functions quickly. Basically, this
    is a convenience wrapper around pandas' read_csv method call."""

    def __init__(self, sep=None, columns=None, column_names=None, encoding=None, engine='python', comment=None,
                 skip=None):
        """Set the parameters to fit the shape of the input data to load."""
        self.config = {'sep': sep, 'usecols': columns, 'names': column_names, 'encoding': encoding,
                       'engine': engine, 'comment': comment, 'skiprows': skip}

    def build(self) -> Callable[[str], pd.DataFrame]:
        """Removes all None values from the configuration, passes it as argument to partial and returns a modified
        verion of pandas' read_csv method. The returned function should only take the file name as argument.

        :return: an extractor function
        """
        final_config = {key: value for key, value in self.config.items() if value is not None}
        return partial(pd.read_csv, **final_config)


def provide_spectrum_constructor(x: str, y: str, **kwargs):
    """
    A convenience function to provide pre-configured constructors for BaseSpectrum objects with custom x and y units
    and data columns to use.

    :param x: name of the column of the extraced data set thet shall be set as x
    :param y: name of the column of the extraced data set thet shall be set as y
    :param kwargs: configuration parameters, typcialle 'x_unit' and 'y_unit'
    :return:
    """

    def spectrum_constructor(name, data, creation_time):
        attributes = kwargs
        attributes['x'] = data[x].to_numpy()
        attributes['y'] = data[y].to_numpy()
        attributes['timestamp'] = creation_time
        return BaseSpectrum(name=name, **attributes)

    return spectrum_constructor


# SFG
def sfg_extractor(file) -> pd.DataFrame:
    """A function extracting the measurement data from a SFG spectral file """
    col_names = ['wavenumbers', 'sfg', 'ir', 'vis']
    return pd.read_csv(file, sep="\t", usecols=[0, 1, 3, 4], names=col_names, encoding='utf8', engine='c')


def sfg_constructor(name, data, creation_time) -> SfgSpectrum:
    """Constructor function for the actual SFG object"""
    meta = {"name": name, "creation_time": creation_time}
    return SfgSpectrum(data['wavenumbers'], data['sfg'], data['ir'], data['vis'], meta)


# LT
def lt_extractor(file) -> pd.DataFrame:
    col_names = ["time", "area", "apm", "surface_pressure"]
    return pd.read_csv(file, comment="#", sep='\t', usecols=[1, 2, 3, 4], names=col_names, engine="c")


def lt_constructor(name, data, creation_time) -> LtIsotherm:
    return LtIsotherm(name, creation_time, data["time"], data["area"], data["apm"], data["surface_pressure"])


# PD
class _PdConfig:
    col_names = ("2theta", "intensity")
    pd_extractor = ExtractorFactory(column_names=col_names, engine='python', sep='\s+').build()

    config = {'x_unit': 'diffraction angle/ 2$\Theta$', 'y_unit': 'intensity/counts'}
    pd_constructor = provide_spectrum_constructor('2theta', 'intensity', **config)
