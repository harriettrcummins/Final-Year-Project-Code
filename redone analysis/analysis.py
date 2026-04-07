from fermipy.gtanalysis import GTAnalysis
import matplotlib
from astropy.table import Table
from time import time
import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from LikelihoodState import LikelihoodState

matplotlib.use('agg')

start = time()


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

def print_green(msg): print(bcolors.OKGREEN + msg + bcolors.ENDC)


def print_yellow(msg): print(bcolors.WARNING + msg + bcolors.ENDC)

radius = 5.0
source = '4FGL J1713.5-3945e'

def free_sources(gta):
    gta.free_sources(free=False)
    gta.free_sources(minmax_ts=[10, None], pars=['norm', 'Prefactor'])
    gta.free_sources(distance=radius, pars=['norm', 'Prefactor', 'alpha', 'beta', 'Index', 'IndexS', 'ExpfactorS'])
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    return

def print_sources(gta, sources):
    print_green("Sources info:")
    if isinstance(sources, str): sources = [sources]
    for src in sources:
        try:
            print()
            print(gta.roi[src])
        except:
            print_yellow(f"{src} not in ROI — skipping")
    return

def print_all_models(gta):
    gta.print_model()
    gta.print_roi()
    gta.print_params()
    return

# Setup GTAnalysis
gta = GTAnalysis('SMC_data3_catupdate/config.yaml', logging={'verbosity': 3})
gta.setup()

# Optimize 3 times
print_green("Starting optimization...")
gta.optimize(skip=[source])
gta.optimize(skip=[source])
gta.optimize(skip=[source])
print_green("Optimization complete.")
gta.write_roi('1-roi_after_optimization')


free_sources(gta)
gta.fit(min_fit_quality=3, optimizer='NEWMINUIT')
gta.write_roi('2-roi_after_fit')
print_green("Initial fit complete.")
print_all_models(gta)
print_sources(gta, source)

# Residual and TS maps
model = {'SpatialModel': 'PointSource', 'Index': 2.0, 'SpectrumType': 'PowerLaw'}
gta.residmap('2-roi_after_fit', model=model, make_plots=True)
gta.tsmap('2-roi_after_fit', model=model, make_plots=True, multithread=True)
gta.write_model_map(model_name='2-roi_after_fit')

gta.fit(min_fit_quality=3, optimizer='NEWMINUIT')
gta.write_roi('2.1-roi_after_fit')

# SED
print_green("Creating SED...")
sed = gta.sed(source, outfile='2.1-roi_after_fit_sed.fits',
              bin_index=2, loge_bins = [2.7, 3.0, 3.5, 4.0, 4.5, 5.0, 5.3, 5.6],
              make_plots=True)
print(sed)

# View SED table
sed_table = Table.read('SMC_data3_catupdate/2.1-roi_after_fit_sed.fits')
print_green("SED Table:")
print(sed_table)

# Remove source and produce residual maps
gta.delete_source(source)
gta.residmap('2-roi_after_fit_source_removed', model=model, make_plots=True)
gta.tsmap('2-roi_after_fit_source_removed', model=model, make_plots=True, multithread=True)

# Energy-dependent maps
gta.residmap('2-roi_after_fit_source_removed_below100GeV', model=model, make_plots=True, loge_bounds=[3, 5])
gta.tsmap('2-roi_after_fit_source_removed_below100GeV', model=model, make_plots=True, loge_bounds=[3, 5], multithread=True)

gta.residmap('2-roi_after_fit_source_removed_above100GeV', model=model, make_plots=True, loge_bounds=[5, None])
gta.tsmap('2-roi_after_fit_source_removed_above100GeV', model=model, make_plots=True, loge_bounds=[5, None], multithread=True)

# Timing
duration_min = round((time() - start) / 60, 2)
print_green(f"Duration: {duration_min} minutes")
print_green("---- DONE ----")
