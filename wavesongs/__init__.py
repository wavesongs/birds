""" Wavesongs package"""
from warnings import filterwarnings
# warnings.simplefilter('once')
filterwarnings('ignore')
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

from .objects.syllable import Syllable
from .objects.song import Song
from .model.bird import motor_gestures, alpha_beta
