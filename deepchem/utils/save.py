#################################################################
# save.py is out of date. You should not import any functions from here.
#################################################################

# flake8: noqa
import logging
logger = logging.getLogger(__name__)
logger.warning("deepchem.utilities.save has been deprecated.\n"
               "The utilities in save.py are moved to deepchem.utilities.data_utils"
               " or deepchem.utilities.genomics_utils.")
from deepchem.utils.data_utils import *
from deepchem.utils.genomics_utils import *
