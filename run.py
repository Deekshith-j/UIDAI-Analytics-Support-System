import sys
import os

# --- Monkeypatch imghdr for Python 3.13 ---
# This must be done BEFORE importing streamlit, as streamlit/altair dependencies might import it at top level.
try:
    import imghdr
except ImportError:
    import types
    imghdr_module = types.ModuleType('imghdr')
    
    def what(file, h=None):
        return None
        
    imghdr_module.what = what
    sys.modules['imghdr'] = imghdr_module
# ------------------------------------------

import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "dashboard/app.py"]
    sys.exit(stcli.main())
