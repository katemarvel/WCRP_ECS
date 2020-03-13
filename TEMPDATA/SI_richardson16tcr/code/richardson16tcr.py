"""
Reconciled climate response estimates from climate models and the energy budget 
of Earth

Published by Nature Climate Change (accepted 2016-05-10)

Author: Mark Richardson, markr@jpl.nasa.gov
Copyright (c) 2016 California Institute of Technology, government sponsorship
acknowledged

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

Run with Python 2.7.9 on an x86_64 GNU/Linux machine. Python can be downloaded
from https://www.python.org/downloads/ or see http://www.SciPy.org/install.html 
for guides on how to obtain science-oriented distributions for Windows, Mac OS
and Linux.

Dependencies: os, np, matplotlib.pyplot, scipy.stats, netCDF4m, datetime,
itertools, mpl_toolkits.basemap, scipy.optimize

DESCRIPTION:
Uses classes and functions called from the associated module.py to generate
results and figures from Richardson et al. (2016) 'Reconciled climate response 
estimates from climate models and the energy budget of Earth'. See README for
description of directory structure.
"""

import richardson16tcr_module as mod
import numpy as np
import itertools
import os

# generate dat object containing all forcing and temperature series
dat = mod.annual_blended_series()

# trueTCR dictionary
with open('../data/TCR/IPCC_TCR.txt','r') as f:
	trueTCR = { line.split('\t')[0] : float(line.strip().split('\t')[1]) for 	
				line in f }

# obtain lists of models for (H)istorical-RCP8.5 and (T)ransient (1pctCO2) 
# simulations, and for models where we have (t)rue TCR, individual (F)orcing 
# series and (joint) where forcing and true TCR are both available.
Hmodels = list( set(dat.hist_rcp85['Had4'].keys()) & 
		 	    set(dat.hist_rcp85['Txax'].keys()) )
Tmodels = list( set(dat._1pctCO2['Had4'].keys()) & 
				set(dat._1pctCO2['Txax'].keys()) &
				set(Hmodels))
tmodels = trueTCR.keys()
Fmodels = dat.RF['allmodels']['models']
jointmodels = [ x for x in tmodels if x in Fmodels ]

# CMIP5 global mean temperature change maps 
dT = mod.ensemble_dT()

#------------------------------------------------------------------------------#
# Produce text files containing main TCR results.
#------------------------------------------------------------------------------#

# 1) canonical and energy-budget TCR for 1pctCO2 runs
calcs = [ mod.simpleTCR, mod.diffTCR, mod.oneboxTCR, mod.trendTCR ]
baselines = [ [0,0], [0,0], [0,0], [0,0] ]
starts = [ 60, 65, 0, 0 ] 
ends = [  79, 74, 70, 40 ]
outfiles = [ '../data/TCR/all/%sTCR.txt' % x for x in  [ 'true', '1pct_diff', 
												 '1pct_onebox', '1pct_trend'] ]

# count through each calculation and assign periods to match, from above the
# values are: simpleTCR is year 60-79 avg minus year 0, difference is year
# yr 65-74 minus 0, onebox is calculated over years 0-70, trend over 0-40.
for a in range(len(calcs)):
	mod.writeTCR(dat=dat,scenario='1pctCO2',calc=calcs[a],obsRFname=None,
				 modRFname='1pctCO2_otto',baseline=baselines[a],start=starts[a],
				 end=ends[a],models=Tmodels,outfile=outfiles[a],use_obs=False,
				 use_index=True)

# 2) energy-budget TCR for all historical_RCP8.5 runs
calcs = [ mod.diffTCR, mod.oneboxTCR, mod.trendTCR ]
baselines = [ [1861,1880], [0,0], [0,0] ]
starts = [ 2000, 1861, 1970 ]
ends = [  2009, 2010, 2010 ]
outfiles = [ '../data/TCR/all/allhist_%sTCR.txt' % x for x in  [ 'diff',
															'onebox', 'trend'] ]

for a in range(len(calcs)):
	mod.writeTCRmodelF(dat=dat,scenario='hist_rcp85',calc=calcs[a],
				 modRFname='forster85',obsRFname='otto',baseline=baselines[a],
			  	 start=starts[a],end=ends[a],models=Hmodels,outfile=outfiles[a],
				 use_obs=True,use_index=False)


#------------------------------------------------------------------------------#
# Calculate statistics of results, supplementary tables.
#------------------------------------------------------------------------------#

# Repeatedly used input variables
TCRfiles = [ '../data/TCR/all/allhist_%sTCR.txt' % x for x in  [ 'diff',
															'onebox', 'trend'] ]
TCRfiles1pct = [ '../data/TCR/all/1pct_%sTCR.txt' % x for x in  [ 'diff',
															'onebox', 'trend'] ]
EBMs = ['Difference','One-box','Trend']

# 1) Table S6, necessary for Table S1. TCR from different baseline and 
#			   reference periods applied to HadCRUT4.
baselines = [ [1861,1880], [1880,1899], [1900,1919] ]
starts = [ 1970,1970,1980,1990,2000 ]
ends = [ 2009,1979,1989,1999,2009 ]

mod.write_diff_tsens_table(dat,baselines,starts,ends,models=Hmodels,
						   outfile='../data/results/diff_tsens_table.txt')

# 2) Table S1, TCR and bias PDF values for different ending years
mod.write_decadal_PDFs(dat,outfile='../data/results/decadal_TCR_PDFs.txt')

# 3) Table S2, mean, median difference from energy-budget calculation to true
mod.EBM_performance_check('../data/TCR/IPCC_TCR.txt',TCRfiles,EBMs,
						  '../data/results/EBM_bias.txt',jointmodels)

# 4) Table S3, percentile at which observations fall for each energy-budget calc
#			   and CMIP5 temperature reconstruction
mod.write_ensemble_pctile(TCRfiles,EBMs,
						  outfile='../data/results/ensemble_TCR_pctile.txt')

# 4) Table S4, using r1i1p1 simulations only
mod.write_ensemble_pctile(TCRfiles,EBMs,sims=['r1i1p1'],
					   outfile='../data/results/ensemble_TCR_pctile_r1i1p1.txt')

# 5) Table S5, relationships between different TCR EBM calculations
mod.write_TCR_correction_stats(TCRfiles,EBMs,skip_header=4,
							  outfile='../data/results/EBMhist_corrections.txt')

# NOTE: Table S6 generate above

# 6) Table S7, ratio of tas-only to blended-masked temperature change for each 
#	 		   model's r1i1p1 simulation.
mod.make_temperature_ratio_table(dat,Hmodels,
								 saveloc='../data/results/tas_bm_ratio.txt')

# 7) Table S8, contribution of each latitude band to temperature change reported
#	 		   1861--1880 to 2000--2009 globally and with 2 masks. S8 
#			   constructed from these data.
mod.masked_zonal_sums(dT,saveloc='../data/results/zonal_mask_Tbias.txt')

# 8) Table S9 column 3, interannual variability of each model from piControl
mod.cmip_natvar(piC_dir = '../data/piControl/tas/',
				outfile = '../data/results/CMIP5_natvar.txt')

# 9) Table S10, list of simulations used in historical-RCP8.5 calculations
mod.write_model_list(Hmodels,Tmodels,Hfileout='../data/results/hist85_sims.txt',
				   Tfileout='../data/results/1pctCO2_sims.txt')

# NOTE: Table S11 is made by combining elements from Tables S2, S5

# 10) Table S12, Table S13 using model sample then Gaussian for correction.
mod.final_TCR_ranges(dat,'trend',TCRfiles[0],
					 '../data/results/T_F_sensitivity_trend.txt')
mod.final_TCR_ranges(dat,'sample',TCRfiles[0],
					 '../data/results/T_F_sensitivity_sample.txt')

#------------------------------------------------------------------------------#
# Main paper figures.
#------------------------------------------------------------------------------#

# 1) Figure 1. Median temperature series for each temperature reconstruction
#	 and differences between them
mod.plot_scenario_mediandiff(dat,Hmodels,saveloc='../figures/Figure1.pdf')

# 2) Figure 2. TCR calculated for HadCRUT4 and histograms of tas-only and
#	 blended-masked CMIP5 simulated values.
mod.TCR_histograms_main(saveloc='../figures/Figure2.pdf')

# 3) Figure 3. Relationship between TCR calculated using difference method for
#	 each temperature reconstruction.
mod.plot_TCR_relationship('../data/TCR/all/allhist_diffTCR.txt',skip_header=4,
						  axlims=[0.0,3.0],saveloc='../figures/Figure3.pdf',
						  headers='',figsize=(6,6))

# 4) Figure 4. 5--95 % ranges and multi-model mean or best estimate of TCR for
#	 CMIP5, Otto et al. and bias-corrected Otto et al.
mod.plot_ranges(saveloc='../figures/Figure4.pdf')


#------------------------------------------------------------------------------#
# Supplementary figures.
#------------------------------------------------------------------------------#

# 1) Figure S1. TCR PDFs for blended-masked and bias-corrected tas-only for each 
#	 ending decade
mod.plot_decadal_TCR_PDFs(saveloc='../figures/SI_Figure1.pdf')

# 2) Figure S2. Histograms of TCR calculated from CMIP5 using each EBM method
#	 for tas-only and blended-masked, versus HadCRUT4 observed
mod.TCR_histograms(saveloc='../figures/SI_Figure2.pdf')

# 3) Figure S3. Relationship between TCR derived from historical-RCP8.5 for
#	 each EBM method using tas-only versus blended and blended-masked
TCRfiles = ['../data/TCR/all/allhist_%sTCR.txt' % x for x in  ['diff', 'onebox', 
															 'trend'] ]
headers = [ 'Difference','One-box','Trend' ]

mod.plot_TCR_relationship(TCRfiles,skip_header=4,axlims=[0,3],headers=headers,
						  saveloc='../figures/SI_Figure3.pdf',
						  figsize=(12,5))

# 4) Figure S4. Time dependence of TCR calculated from each EBM method for
#	 historical-RCP8.5 sims for tas-only and blended

# generate data for plot
mod.writeTCR_tsensitivity(dat,'Txax',RF=dat.RF['forster85'],
						  models=Hmodels,modelF=True)
datadir = '../data/TCR/tsensitivity/'
files = [ x for x in os.listdir(datadir) if 'histrcp85_' in x ]
titles = [ '%s %s' % (x,y) for x,y in itertools.product(['Difference','One-box',
									   'Trend'],['tas-only','blended']) ]

mod.plot_tsensitivities('../data/TCR/tsensitivity/',files,titles,nx=2,ny=3,
						xlim=[1980,2100],
						ylim=[0,3],saveloc='../figures/SI_Figure4.pdf')

# 5) Figure S5. Time series of median and 5--95 % range for CMIP5 vs HadCRUT4
#	 for tas-only and blended-masked, from 1861--2014
mod.plot_CMIP5_v_obs(dat,Hmodels,saveloc='../figures/SI_Figure5.pdf')

# 6) Figure S6. Maps of global temperature change seen for each 
#	 period of coverage using CMIP5 ensemble mean
mod.plot_coverage_Tmaps([dT.dtasglob,dT.dtas2000,dT.dtas1861],
						saveloc='../figures/SI_Figure6.pdf')

# 7) Figure S7. Zonal profiles of temperature change seen by
#	 different period coverage masks and globally, absolute and area-weighted.
mod.zonal_dT_profiles(dT,saveloc='../figures/SI_Figure7.pdf')

# 8) Figure S8. Zonal profiles of cumulative contribution to mask
#	 temperature bias.
mod.plot_zonebias_contribution(saveloc='../figures/SI_Figure8.pdf',
						zonebias_dataloc='../data/results/zonal_mask_Tbias.txt')

# 9) Figure S9. Relationship between canonical TCR from tas-only
#	 versus blended and blended-masked in 1pctCO2 simulations
mod.plot_TCR_relationship(TCRfiles='../data/TCR/all/trueTCR.txt',skip_header=1,
						  axlims=[0,2.6],figsize=(6,6),
						  saveloc='../figures/SI_Figure9.pdf',writefit=True)

# 10) Figure S10. PDFs of TCR for blended-masked, tas-only, using
#	  wider F uncertainty to match Lewis & Curry, correction from TCRfile.
mod.plot_TCR_PDFs(dat,TCRfile='../data/TCR/all/allhist_diffTCR.txt',
			  saveloc='../figures/SI_Figure10.pdf',sampleN=1e6)
