class InvalidSpectrumError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class SpectrumIntegrationError(Exception):
    pass


class CoverageCalculationImpossibleError(Exception):
    pass


class SpectrumTypeNotAvailableError(Exception):
    pass


class NoTimestampOfMeasurementSetError(Exception):
    pass

class MaximumPressureCalculationError(Exception):
    pass