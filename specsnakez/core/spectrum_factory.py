from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import Any, Callable, List

from .base_spectrum import BaseSpectrum
from .builtin_loaders import sfg_extractor, sfg_constructor, lt_extractor, lt_constructor, _PdConfig
from .exceptions import SpectrumTypeNotAvailableError


class SpectrumFactoryProvider:
    """A class to generate SpectrumFactories. This is the generic way to introduce new types of spectra to
    the logic of this package. All builtin type of spectra are available at runtime due to an instance of this class
    and to add new types, the underlying config dictionary may be updated."""

    global_factory_config = {
        "sfg": {"extractor": sfg_extractor, "constructor": sfg_constructor},
        "lt": {"extractor": lt_extractor, "constructor": lt_constructor},
        "pd": {'extractor': _PdConfig.pd_extractor, "constructor": _PdConfig.pd_constructor}
    }

    def provide_factory_by_name(self, name: str) -> CustomSpectrumFactory:
        try:
            return CustomSpectrumFactory(**self.global_factory_config[name])
        except KeyError:
            raise SpectrumTypeNotAvailableError(
                "The spectrum type you requested is not available in the current config!")

    def add_new_template(self, name: str, extractor: Callable[[Path], Any],
                         constructor: [[str, Any, datetime.datetime], BaseSpectrum],
                         name_transformer: Callable[[str], str] = None):
        """Add a new template for SpectrumFactories to the configuration. This requires the name of the template,
        an extractor method, a constructor method and an optional name_transformer method.

        :param name: the name the created template configuration
        :param extractor: the function that is called on the file path to extract the raw data
        :param constructor: the function that is called to generate the actual spectrum instance
        :param name_transformer: an optional function to convert the file name in a suitable string for plot legends"""
        self.global_factory_config[name] = {'extractor': extractor, 'constructor': constructor,
                                            'name_transformer': name_transformer}


class CustomSpectrumFactory:

    def __init__(self, extractor: Callable[[Path], Any], constructor: [[str, Any, datetime.datetime], BaseSpectrum],
                 name_transformer: Callable[[str], str] = None):
        """Configure the factory to create spectrum objects from raw measurement data. The extractor is a function that extracts
        the data from the file, the constructor is responsible for the instantiation of the spectrum objects and the
        optional name transformer processes the file name to a representation suitable to appear in a plot legend.

        :param extractor: the function that is called on the file path to extract the raw data
        :param constructor: the function that is called to generate the actual spectrum instance
        :param name_transformer: an optional function to convert the file name in a suitable string for plot legends
        """
        self.extractor = extractor
        self.constructor = constructor
        self.name_transformer = name_transformer

    def build_from_file(self, file: Path) -> BaseSpectrum:
        """The function that transforms the filepath to the spectrum object+

        :param file: the Path of the file to extract
        :return: the result of the call to the constructor attribute
        """
        creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(str(file)))
        data = self.extractor(file)

        if not self.name_transformer:
            name = file.stem
        else:
            name = self.name_transformer(file)

        return self.constructor(name, data, creation_time)

    def build_batch(self, directory: Path, file_ending='*') -> List[BaseSpectrum]:
        """A convenience function to apply the builder to all spectra within the directoy and return a list of spectra.
        Can receive a pattern for the file ending to avoid the wrong files being imported.
        :param directory: the directory in which all files shall be converted to spectra
        :param file_ending: restricts the extraction to files in the directory with a specific file ending
        :return: a list of spectrum instances

        """
        return [self.build_from_file(p) for p in directory.rglob(file_ending)]
