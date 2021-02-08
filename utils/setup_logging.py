import meta.CONSTANTS as see
from meta.utils.BaseExperimentLogger import BaseExperimentLogger


def setup_logging(*args, **kwargs):
    see.logs = BaseExperimentLogger(*args, **kwargs)
    see.logs.make_logdir()


