import datetime
import itertools as ito
import os
from typing import Dict, List

import numpy as np

from specsnakez.core.base_spectrum import BaseSpectrum
from specsnakez.core.langmuir_isotherm import LtIsotherm
from specsnakez.core.sfg_spectrum import SfgSpectrum, DummyPlotter
from specsnakez.orm.base_dtos import SFG
from specsnakez.orm.import_db_controller import DatabaseWizard


class DbInteractor(DatabaseWizard):

    def __init__(self):
        super().__init__()
        self.session.commit()
        self.references = self.get_reference_integrals()

    def get_reference_integrals(self) -> Dict[datetime.date, float]:
        """Get a map of the dates of measurement dates to the respective DPPC integrals"""
        temp = self.session.query(self.measurement_days).all()
        q = {key: list(value)[0].dppc_integral for key, value in ito.groupby(temp, key=lambda x: x.date)}
        return q

    # auxiliary functions

    def fetch_by_specid(self, specid, sfg=True) -> BaseSpectrum:
        """Fetches the SFG spectrum with the id specid from the database and retuns it as an SFG spectrum object."""
        if sfg:
            spec = self.session.query(self.sfg).filter(self.sfg.id == specid).one()
            return self.construct_sfg(spec)
        else:
            lt = self.session.query(self.lt).filter(self.lt.id == specid).one()
            return self.construct_lt(lt)

    def get_spectrum_by_name(self, name) -> SfgSpectrum:
        """Returns the SFG spectrum object for a given file name"""
        temp = self.session.query(self.sfg).filter(self.sfg.name == name).one()
        return self.construct_sfg(temp)

    def get_spectrum_by_property(self, property_, target) -> list[SfgSpectrum]:
        """A convenience function to collect spectra based on properties like surfactant, sensitizer etc."""
        temp = self.session.query(self.regular_sfg). \
            filter(getattr(self.regular_sfg, property_) == target).all()
        out = []
        for item in temp:
            out.append(self.get_spectrum_by_name(item.name))
        return out

    def convert_regular_to_lt(self, reg_lt) -> LtIsotherm:
        """Converts a RegularLt object directly into the Lt object of the spectrum module."""
        lt = self.session.query(self.lt).filter(self.lt.id == reg_lt.ltid).one()
        return DbInteractor.construct_lt(lt)

    def convert_regular_to_sfg(self, reg_sfg) -> SfgSpectrum:
        """Converts a RegularSfg object directly into the Sfg object of the spectrum module.
        It remains the former regular_sfg object as part of the new spectrum's meta attribute
        for access of the metadata stored in the regular_sfg object."""
        sfg = self.session.query(self.sfg).filter(self.sfg.id == reg_sfg.specid).one()
        temp = DbInteractor.construct_sfg(sfg)
        temp.meta["regular"] = reg_sfg
        return temp

    def map_data_to_dates(self, data) -> Dict[datetime.date, List[BaseSpectrum]]:
        """This function maps a list of spectra to their corresponding date of measurement
        """
        dates = {}
        for item in data:
            sampling_date = item.measured_time.date()
            if sampling_date not in dates:
                dates[sampling_date] = [item]
            else:
                dates[sampling_date].append(item)
        return dates

    def origin_preview_date(self, surfacant="NA", out_dir="out", max_size=6):
        temp = self.session.query(self.regular_sfg).filter(self.regular_sfg.surfactant == surfacant).all()
        temp = [self.session.query(self.sfg).filter(self.sfg.id == i.specid).one() for i in temp]
        dates = self.map_data_to_dates(temp)
        for key in dates:
            dir_name = out_dir + "/" + str(key)
            os.mkdir(dir_name)
            sfg_spectra = [self.construct_sfg(i) for i in dates[key]]

            for spec in sfg_spectra:
                df = spec.convert_to_export_dataframe()
                df.to_csv(f'{dir_name}/' + spec.name + ".csv", index=False, sep=";")

            sub_speclist = [sfg_spectra[i:i + max_size] for i in range(0, len(sfg_spectra), max_size)]
            for index, item in enumerate(sub_speclist):
                DummyPlotter(item, save=True, savedir=dir_name, savename=f'preview{index}').plot_all()

    # debugging

    def get_unmapped_be(self, selection, output_directory):
        if selection == "entry":
            temp = self.session.query(self.be_water_samples).filter(self.be_water_samples.sfg_id == None).order_by(
                self.be_water_samples.sampling_date).all()
        elif selection == "spectra":
            temp = self.session.query(self.boknis_eck).filter(self.boknis_eck.table_entry_id == None).order_by(
                self.boknis_eck.sampling_date).all()
        with open(f'{output_directory}/{selection}_debug_log.txt', 'w') as outfile:
            for t in temp:
                outfile.write(str(t) + '\n')

    # essential
    def get_coverage(self, spectrum: SFG) -> float:
        """Calculate the surface coverage of a given SFG spectrum object by dividing its CH integral by the CH integral
        obtained during the corresponding day of measurement and taking the square root."""
        reference = self.get_reference_integrals()[spectrum.measured_time.date()]
        integral = DbInteractor.construct_sfg(spectrum).calculate_ch_integral()
        return np.sqrt(integral / reference)

    @staticmethod
    def construct_lt(or_object) -> LtIsotherm:
        """A function constructing the LT object from the orm declarative class."""
        args = (or_object.name, or_object.measured_time)
        add_args = ["time", "area", "apm", "surface_pressure", "lift_off"]
        add_args = [DbInteractor.to_array(getattr(or_object, i)) for i in add_args]
        l = LtIsotherm(args[0], args[1], *add_args)
        return l
