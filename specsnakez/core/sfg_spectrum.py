from __future__ import annotations
import copy
import csv
import datetime
from typing import List, Tuple, Dict, Any
import logging
import numpy as np
import pandas as pd
import peakutils
from matplotlib import pyplot as plt
from scipy.integrate import simps as sp, trapz as tp

from specsnakez.core.base_spectrum import BaseSpectrum
from specsnakez.core.exceptions import CoverageCalculationImpossibleError

logger = logging.getLogger(__name__)


class SfgSpectrum(BaseSpectrum):
    """The SFG spectrum class is the foundation of all analysis and plotting tools. It contains a class
    SystematicName (or a derived class) which carries most of the metainformation. Besides holding the
    experimental data, it gives access to a variety of functions like normalization, peak picking etc."""

    def __init__(self, wavenumbers: np.ndarray, intensity: np.ndarray, ir_intensity: np.ndarray,
                 vis_intensity: np.ndarray, meta: Dict[str, Any]):
        """

        :param wavenumbers: wavenumber of the tunable IR beam
        :param intensity: recored SFG intenity
        :param ir_intensity: IR reference detector intensity
        :param vis_intensity: VIS reference detector intensity
        :param meta: metadata dictionary, including timestamp of creation
        """
        self.x = wavenumbers[::-1]
        self.raw_intensity = intensity[::-1]
        self.vis_intensity = vis_intensity[::-1]
        self.ir_intensity = ir_intensity[::-1]
        self.meta = meta
        self.y = self.raw_intensity / (self.vis_intensity * self.ir_intensity)
        self.baseline_corrected = None

        self.x_unit = "wavenumber/ cm$^{-1}$"
        self.y_unit = "SFG intensity/ arb. u."
        self.name = self.meta["name"]

        self.regions = None
        self.set_regions()

    def __lt__(self, other: SfgSpectrum) -> bool:
        """Returns true if the current spectrum was measured before SFG2
        :param other: other SFG spectrum to compare to
        """
        return True if self.meta["creation_time"] < other.name.meta["creation_time"] else False

    # spectral data processing and analysis tools
    def integrate_peak(self, x_array: np.ndarray, y_array: np.ndarray) -> float:
        """
        Numpy integration routine for numerical peak integration with the trapezoidal rule.

        :param x_array: the x values (spacing) of the curve to integrate
        :type x_array: array-like
        :param y_array: the y values of the curve to integrate
        :type y_array: array-like
        :return: the area under the x/y curve
        :rtype: float
        """
        try:
            area = sp(y_array, x_array)
            if np.isnan(area):
                area = tp(y_array, x_array)
            return area
        except:
            raise ValueError(f'Integration not possible for {self.name}')

    def root(self) -> np.ndarray:
        """
        :return: the spectrum's normalized intensity
        :rtype: np.ndarray
        """
        return np.sqrt(self.y)

    def yield_maximum(self) -> float:
        """
        :return: maximum intensity value of the spectrum
        :rtype: float
        """
        return np.max(self.y)

    def yield_spectral_range(self):
        """returns a list containing maximum and minimum wavenumer and the number of data points"""
        return [min(self.x), max(self.x), len(self.x)]

    def yield_increment(self) -> List[List[Any], List[Any]]:
        """Calculates stepsize and wavenumbers where the stepsize is changed"""
        borders = []
        stepsize = []
        current = self.x[0]
        currentstep = abs(current - self.x[1])
        borders.append(current)

        for wavenumber in self.x[1:]:
            s = abs(wavenumber - current)
            if s != currentstep:
                stepsize.append(currentstep)
                borders.append(current)
                currentstep = s
                current = wavenumber
            else:
                current = wavenumber
        borders.append(current)
        stepsize.append(s)
        increment = [borders, stepsize]
        return increment

    def yield_wn_length(self):
        return np.max(self.x) - np.min(self.x)

    # info functions

    def drop_ascii(self) -> None:
        """Create an ascii file with the wavenumbers and normalized intensities"""
        with open(self._name + ".csv", "w") as outfile:
            writer = csv.writer(outfile, delimiter=";")
            for i in zip(self.x, self.y):
                writer.writerow((i[0], i[1]))

    def convert_to_export_dataframe(self) -> pd.DataFrame:
        """
        This function returns a Pandas dataframe, suitable for data export to
        origin and similar other programs

        :return: a pandas dataframe with the spectral data
        :rtype: pd.DataFrame
        """
        data = {
            "wavenumbers": self.x,
            "normalized intensity": self.y,
            "raw intensity": self.raw_intensity,
            "IR intensity": self.ir_intensity,
            "VIS intensity": self.vis_intensity
        }
        return pd.DataFrame(data=data)

    # auxiliary function
    def slice_by_borders(self, lower, upper) -> Tuple[int, int]:
        """Takes a high (upper) and a low (lower) reciprocal centimeter value as argument. Returns
        the indices of the wavenumber array of the spectrum that are the borders of this interval."""
        lower_index = np.argmax(self.x >= lower)
        upper_index = np.argmax(self.x >= upper)
        return int(lower_index), int(upper_index)

    def set_regions(self):
        self.regions = {"CH": (int(np.min(self.x)), 3000),
                        "dangling": (3670, 3760),
                        "OH": (3005, 3350), "OH2": (3350, 3670)}

    def split_y_into_halfs(self):
        lower = self.get_xrange_indices(np.min(self.x), 3030)
        upper = self.get_xrange_indices(3030, np.max(self.x))

        # it may happen that extremely small negative numbers appear, causing the square root to fail. Those will be set
        # to zero here. The next problem is that sometimes a division by zero value in the normalization leads to the
        # occurrence of NAN values. They will be also set to zero
        negative_removed = SfgSpectrum.remove_nan_and_negative(self.y)

        left = negative_removed[lower[0]:lower[1] + 1]
        right = negative_removed[upper[0] + 1:upper[1] + 1]
        return left, right

    # baseline correction
    def make_ch_baseline(self, debug=False):

        if np.min(self.x) > 2800:
            left = self.slice_by_borders(np.min(self.x), 2815)
        else:
            left = self.slice_by_borders(np.min(self.x), 2800)

        if np.max(self.x) >= 3030:
            right = self.slice_by_borders(3000, 3030)
        else:
            right = self.slice_by_borders(self.x[-4], self.x[-1])

        left_x = self.x[left[0]:left[1] + 1]
        left_y = self.y[left[0]:left[1] + 1]

        right_x = self.x[right[0]:right[1] + 1]
        right_y = self.y[right[0]:right[1] + 1]

        slope = (np.average(right_y) - np.average(left_y)) / \
                (np.average(right_x) - np.average(left_x))

        intercept = np.average(left_y) - slope * np.average(left_x)

        if debug:
            logger.debug(f'intercept: {intercept}, slope: {slope}, left:{left}, right: {right}')

        def baseline(x):
            return x * slope + intercept

        return baseline

    def correct_baseline(self):
        """If the baseline correction was not performed, set the instance attribute baseline_corrected to baseline-corrected
        array of the initial y value."""
        if self.baseline_corrected is not None:
            return self.baseline_corrected

        else:
            # if the baseline correction already was performed, return immediately
            temp = copy.deepcopy(self.y)

            if np.max(self.x) >= 3000:
                borders = (2750, 3000)
            else:
                borders = (2750, np.max(self.x))

            func = self.make_ch_baseline()

            xvals = self.x.copy()
            corr = func(xvals)

            # ensure that only the region of the spec defined in the borders is used
            # xvals is a vector being 0 everywhere except in the to-correct area where it
            # is 1 so that xvals*corr yields nonzero in the defined regions only
            np.putmask(xvals, xvals < borders[0], 0)
            np.putmask(xvals, xvals > borders[1], 0)
            np.putmask(xvals, xvals != 0, 1)

            corr *= xvals

            # apply the correction
            temp -= corr

            self.baseline_corrected = temp
            return temp

    def full_baseline_correction(self):
        left, right = self.split_y_into_halfs()
        left_base = peakutils.baseline(left, deg=1)
        right_base = peakutils.baseline(right, deg=1)
        return np.concatenate([left - left_base, right - right_base])

    def root_baseline_correction(self, square=False):
        left, right = self.split_y_into_halfs()
        left = SfgSpectrum.remove_nan_and_negative(np.sqrt(left))
        right = SfgSpectrum.remove_nan_and_negative(np.sqrt(right))

        left_base = peakutils.baseline(left, deg=1)
        right_base = peakutils.baseline(right, deg=1)
        # todo: the general shape concat(left - left_base, right - right_base) may be abstracted
        return np.concatenate([left - left_base, right - right_base])

    # integration
    def calculate_ch_integral(self, baseline_function=None) -> float:
        """Calculate the integral of the spectrum in the range of 2750-3000 wavenumbers. Use the switch 'old_baseline' :exception
        to choose between the old and new style of baseline correction"""
        if baseline_function is None:
            baseline_function = self.full_baseline_correction

        # check upper border
        upper = 3000 if max(self.x) >= 3000 else max(self.x)

        # check lower border
        lower = 2750 if min(self.x) <= 2750 else min(self.x)

        borders = self.slice_by_borders(lower, upper)
        x_array = self.x[borders[0]:borders[1] + 1]
        # todo at this stage the baseline algorithm has to be chosen
        try:
            y_array = baseline_function()[borders[0]:borders[1] + 1]
        except ValueError:
            y_array = self.correct_baseline()[borders[0]:borders[1] + 1]
            logger.error(f'{self.name} in borders of {self.yield_spectral_range()} fails in normal baseline routine.'
                         f'Falling back to old procedure.')

        integral = self.integrate_peak(x_array, y_array)
        return integral

    def calc_region_integral(self, region):
        borders = self.regions[region]
        borders = self.slice_by_borders(borders[0], borders[1])
        x_array = self.x[borders[0]:borders[1] + 1]
        y_array = self.y[borders[0]:borders[1] + 1]
        try:
            integral = self.integrate_peak(x_array, y_array)
            if np.isnan(integral):
                logger.warning(f'x: {x_array}, y: {y_array}')
            return integral

        except ValueError:
            logger.warning(f'Integration not possible in {self.name} in region{region}')
            return np.nan

    @staticmethod
    def remove_nan_and_negative(array, replacement=0):
        mask = (array < 0) | (~np.isfinite(array))
        return np.where(mask, replacement, array)


# todo: check on instantiation if a spectrum has a suitable reference, exclude it
class SfgAverager:
    # todo: reference_part function needs refactoring for readability, it's not pythonic
    """This class takes a list of SFG spectra and generates an average spectrum of them by interpolation and
    averaging. It is possible to pass a dictionary of date:dppc_integral key-value-pairs in order to calculate
    the coverage."""

    def __init__(self, spectra: List[SfgSpectrum], references: Dict[datetime.date, float] = None,
                 enforce_scale: bool = False, name: str = "default",
                 baseline: bool = False):
        """

        :param spectra: list of SFG spectra to average
        :param references: a dictionary of average DPPC integrals mapped to the corresponding dates of measurement
        :param enforce_scale: if True, the resulting average spectrum is enfored to have a certain scale
        :param name: a name to add as plot title of file ending for export
        :param baseline: if enabled, baseline correction ois applied to the resulting average spectrum
        """
        self.spectra = spectra
        self.references = references
        self.enforce_scale = enforce_scale
        self.name = name
        self.day_counter = {}
        self.average_spectrum = None
        self.integral = None
        self.coverage = None
        self.total = None

        if len(self.spectra) == 0:
            logger.warning("Warning: zero spectra to average in SfgAverager!")

        else:
            self.average_spectrum = self.average_spectra()
            self.integral = self.average_spectrum.calculate_ch_integral()
            try:
                self.coverage = self.calc_coverage()
            except CoverageCalculationImpossibleError:
                self.coverage = None

    def average_spectra(self) -> AverageSpectrum:
        """Function performing the averaging: it ensures that all spectra are interpolated to have the same shape,
        then they are averaged. A AverageSpectrum  object is constructed and returned."""
        to_average = []

        # sort spectra by length of the wavenumber array (lambda)
        if self.enforce_scale is False:
            self.spectra.sort(key=lambda x: x.yield_wn_length(), reverse=True)
            root_x_scale = self.spectra[0].x
        else:
            root_x_scale = SfgAverager.enforce_base()

        # get y values by interpolation and collect the y values in a list
        # collect the dates of measurement for DPPC referencing
        for item in self.spectra:

            date = item.meta["time"].date()

            if date not in self.day_counter:
                self.day_counter[date] = 1
            else:
                self.day_counter[date] += 1

            new_intensity = np.interp(root_x_scale, item.x, item.y)
            mask = (root_x_scale > np.max(item.x)) | (root_x_scale < np.min(item.x))
            new_intensity[mask] = np.nan
            to_average.append(new_intensity)

        to_average = np.array(to_average)
        average = np.nanmean(to_average, axis=0)
        std = np.nanstd(to_average, axis=0)

        # prepare meta data for average spectrum
        if self.name == "default":
            newname = self.spectra[0].name + "baseAV"
        else:
            newname = self.name
        in_new = [n.name for n in self.spectra]
        s_meta = {"name": newname, "made_from": in_new, "std": std}

        return AverageSpectrum(root_x_scale, average, s_meta)

    def calc_reference_part(self) -> float:
        """Calculate the participation of each DPPC references. This is important if the spectra to average are
        measured on different sampling days. If, for example,  5 samples are to average and 3 of them are measured
        on one day, 2 on another, the final coverage is calculated by dividing the AveragedSpectrum integral by the
        weighted sum of the DPPC integrals of the corresponding days, in our example

        .. math::

            \\frac{2}{5} \\cdot r_1 + \\frac{3}{5} \\cdot r_2

        """

        spec_number = len(self.spectra)
        total = 0
        logger.info(f'Start of the reference calculating section:')

        for date in self.day_counter:
            # divide by total number of spectra in the average
            logger.info(f'date {date} divided by the total spec number to average {spec_number}')
        self.day_counter[date] /= spec_number

        # multiply the weighting factor by the integral of the day
        try:
            self.day_counter[date] *= self.references[date]
            total += self.day_counter[date]
            logger.info(f"""Now multiplying the factor {self.day_counter[date]} 
                by the reference integral {self.references[date]} resulting in a dppc_factor of {total:-4f}""")
        except KeyError:
            logger.error(f'Error: no suitable DPPC reference found four date {date}')

        self.total = total
        return total

    def calc_coverage(self, baseline_function=None) -> float:
        """A convenience function  to calculate the surface coverage. A custom function to yield a baseline-corrected
        value of the spectrum's y data may be provided."""
        if self.references is not None:
            dppc_factor = self.calc_reference_part()
            integral = self.average_spectrum.calculate_ch_integral(baseline_function)
            coverage = np.sqrt(integral / dppc_factor)
            logger.info(f'Calculating coverage: integral = {integral:.4f}, '
                        f'dppc_factor = {dppc_factor:.4f}, coverage = {coverage:.4f}')
            return coverage

        else:
            raise CoverageCalculationImpossibleError(
                f'Coverage not available for reference samples, integral is {self.integral}!')

    @staticmethod
    def enforce_base() -> np.ndarray:
        reg1 = np.arange(2750, 3055, 5)
        reg2 = np.arange(3050, 3670, 20)
        reg3 = np.arange(3650, 3845, 5)
        new = np.concatenate((reg1, reg2, reg3), axis=None)
        return new


class AverageSpectrum(SfgSpectrum):
    """The class resulting from averaging multiple spectra. It is aware of the spectra it was generated
    from."""

    def __init__(self, wavenumbers: np.ndarray, intensities: np.ndarray, meta: dict[str, Any]):
        self.x = wavenumbers
        self.y = intensities
        self.x_unit = "wavenumber/ cm$^{-1}$"
        self.y_unit = "SFG intensity/ arb. u."
        self.name = meta["name"]

        self.meta = meta
        self.baseline_corrected = None
        self.regions = None

        # ensure nan-values in intensity and their corresponding wavenumbers are removed
        mask = np.isfinite(self.y)
        self.y = self.y[mask]
        self.x = self.x[mask]
        super().set_regions()


class DummyPlotter:
    """A test class to monitor the interaction of the subclasses of AbstractSpectrum with plotting routines."""

    # todo: remove from module
    def __init__(self, speclist, save=False, savedir="", savename="Default", special=None):
        self.speclist = speclist
        self.special = special
        self.save = save
        self.savedir = savedir
        self.savename = savename

    def plot_all(self, base=False, marker=True):
        for spectrum in self.speclist:

            if base is True:
                if isinstance(spectrum, AverageSpectrum):
                    func = spectrum.make_ch_baseline()
                    testx = np.linspace(2750, 3000, 1000)
                    testy = func(testx)
                    plt.plot(testx, testy, color="black")
                    integral = spectrum.calculate_ch_integral()
                    plt.title(str(round(integral, 6)))

            if self.special is None:
                if marker:
                    plt.plot(spectrum.x, spectrum.y, label=spectrum.name, marker="^", linestyle="-")
                else:
                    plt.plot(spectrum.x, spectrum.y, label=spectrum.name)

            else:
                if self.special not in spectrum.name:
                    plt.plot(spectrum.x, spectrum.y, label=spectrum.name, marker="^", alpha=0.3)
                else:
                    plt.plot(spectrum.x, spectrum.y, label=spectrum.name, marker="^", linestyle="-", color="r")

        plt.xlabel(spectrum.x_unit)
        plt.ylabel(spectrum.y_unit)
        plt.minorticks_on()
        plt.legend()

        if self.save is False:
            plt.show()

        else:
            path = self.savedir + "/" + self.savename + ".png"
            plt.savefig(path)
            plt.close()
