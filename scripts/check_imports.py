"""Import every module given on the command line (catches missing optional deps)."""

import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    failed = False
    for path in sys.argv[1:]:
        try:
            SourceFileLoader("x", path).load_module()
        except Exception:
            failed = True
            print(path)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
