import re

__version__ = "0.0.1dev0"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re

