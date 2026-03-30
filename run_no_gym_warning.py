"""
Run a module with the gym deprecation warning suppressed (so it runs first).

Usage (equivalent to `python -m gto.generate_tables --board-sample 50 ...` but no gym message):

    python run_no_gym_warning.py gto.generate_tables --board-sample 50 --board-sample-preflop 100 ...

Or set once in the shell and use -m as usual:

    set PYTHONWARNINGS=ignore::UserWarning
    python -m gto.generate_tables --board-sample 50 ...
"""
import runpy
import sys
import warnings

# Suppress gym unmaintained warning before any other imports
warnings.filterwarnings("ignore", message=".*[Gg]ym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*[Gg]ymnasium.*")
warnings.filterwarnings("ignore", message=".*upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

module = sys.argv[1]
# Pass remaining args to the module (run_module sets argv[0] to the module path)
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__")
