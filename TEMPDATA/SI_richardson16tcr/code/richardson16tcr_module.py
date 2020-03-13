""" Reconciled climate response estimates from climate models and the energy budget 
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

This contains classes and functions used in main code, see associated python
script for details.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
import netCDF4 as nc
import datetime as dt
import itertools
from mpl_toolkits.basemap import Basemap,cm
from scipy.optimize import curve_fit

#------------------------------------------------------------------------------#
# Classes
#------------------------------------------------------------------------------#

class annual_blended_series():
	""" Class containing main data time series used in analysis.
	
	Contains:
		Tobs : dictionary of observed temperature series at annual resolution, 
			   where each observational series is stored as a dictionary with 
			   entries 't' and 'T'
		RF : dictionary of input forcing series, each series as dictionary with
			 entries 't' and 'F'
		_1pctCO2 : CMIP5 model temperature series in from 1pctCO2 temperature
				   series with entries 'Txax' (global air), 'Had4' (HadCRUT4
				   coverage). Each model is stored as a dictionary within this, 
				   and contains 't', 'T' (blended) and 'T0' (unblended, i.e.
				   near-surface air temperature only)
		hist_rcp85 : CMIP5 simulated temperature series from historical-RCP8.5
					 simulations. Formatted as in _1pctCO2 except that HadCRUT4
					 mask matches historical HadCRUT4 coverage.
	"""
	
	def __init__(self,options_1pct=['Had4','Txax'],
				 options_hist85=['Had4','Txax'],rootdir='/Users/kmarvel/Google Drive/WCRP_ECS/TEMPDATA/SI_richardson16tcr/data/'):
		#----------------------------------------------------------------------#
		# (1) Load the model output
		#----------------------------------------------------------------------#
			
		self._1pctCO2 = dict((option,{}) for option in options_1pct)
		self.hist_rcp85 = dict((option,{}) for option in options_hist85)
		
		for option in options_1pct:
			f_1pct = os.listdir(rootdir+'1pctCO2/'+option)
			self._1pctCO2[option] = dict( (model[:-5],{}) for model in f_1pct)
			
			for model in f_1pct:
				data = np.genfromtxt(rootdir+'1pctCO2/'+option+'/'+model)
				varnames = self.return_varnames(data)
				
				for a in range(len(varnames)):
					self._1pctCO2[option][model[:-5]][varnames[a]] = data[:,a]
					if 'T' in varnames[a]:
						self._1pctCO2[option][model[:-5]][varnames[a]] -= \
							   self._1pctCO2[option][model[:-5]][varnames[a]][0]
		
		for option in options_hist85:
			f_hist85 = os.listdir(rootdir+'hist_rcp85/'+option)
			self.hist_rcp85[option] = dict((model[:-5],{}) for model in 
										   f_hist85)
			
			for model in f_hist85:
				data = np.genfromtxt(rootdir+'hist_rcp85/'+option+'/'+model)
				varnames = self.return_varnames(data)
				
				for a in range(len(varnames)):
					self.hist_rcp85[option][model[:-5]][varnames[a]] = data[:,a]
		
		#----------------------------------------------------------------------#
		# (2) Observational temperature series
		#----------------------------------------------------------------------#
		Tserieslist = [ x.replace('.txt','') for x in 
						os.listdir(rootdir+'observations/') ]
		self.Tobs = dict( (key,{}) for key in Tserieslist)
						
		for Tseries in Tserieslist:
			data = np.genfromtxt(rootdir+'observations/'+Tseries+'.txt')
			
			self.Tobs[Tseries]['t'] = data[:,0]
			self.Tobs[Tseries]['T'] = data[:,1]
		
		#----------------------------------------------------------------------#
		# (3) Forcing time series
		#----------------------------------------------------------------------#
		forcinglist = ['otto','forster85','allmodels']
		skip_headers = [0,0,0,1]
		self.RF = dict((key,{}) for key in forcinglist)
		
		for a in range(len(forcinglist)):
			forcing,skip_header = forcinglist[a],skip_headers[a]
			data = np.genfromtxt(rootdir+'forcing/'+forcing+'.txt',
								 skip_header=skip_header)
			
			self.RF[forcing]['t'] = data[:,0].astype('int') # use integer years
			if data.shape[1] > 2:
				self.RF[forcing]['F'] = data[:,1:]
			else:
				self.RF[forcing]['F'] = data[:,1]
		
		# write list of models in order for allmodels
		with open(rootdir+'forcing/allmodels.txt','r') as f:
			self.RF['allmodels']['models'] = \
								np.array( f.readline().strip().split('\t')[1:] )
		
		# generate 1pctCO2 simulated forcing, assuming 3.7 and 3.44 forcing
		# for doubled CO2.
		t = np.arange(0,140)
		C = np.array([ 1.01**x for x in t ])
		
		self.RF['1pctCO2_otto'] = { 't':t , 'F':4.97*np.log(C) }
		self.RF['1pctCO2_myhre'] = { 't':t,'F':5.35*np.log(C) }
	
	#----------------------------------------------------------------------#
	# Function called during init
	#----------------------------------------------------------------------#	
	def return_varnames(self,data):
		""" Look at a data file and determine its shape. If more than 2 columns,
		then is in Kevin Cowtan's ncblendmask.py output format. Otherwise it's 
		just time and temperature. """
		if np.shape(data)[1] > 2:
			return ['t','T0','T']
		else:
			return ['t','T']


class ensemble_dT():
	""" CMIP5 ensemble mean temperature changes with different masks to allow
	attribution of masking effect to different regions.
	
	Contains:
		cvgmasks : coverage maps 'all' (global monthly coverage for one decade)
				   '2000-2009' (2000--2009 monthly coverage)
				   '1861-1880' (1861--1880 monthly coverage)
		dtas : map of ensemble-mean temperature change from 1861--1880 to 
			   2000--2009 in CMIP5 for air temperatures
		dxax : ensemble-mean temperature change based on anomaly calculations
			   over same period
	"""	
	
	def __init__(self,tasfile=None,xaxfile=None,CRUfile=None,SSTfile=None):
		"""
		"""
		
		# if files not provided, use default
		if tasfile == None:
			tasfile='../data/tas-ensmean-5x5.nc'
		if xaxfile == None:
			xaxfile='../data/blend-xax-ensmean-5x5.nc'
		if CRUfile == None:
			CRUfile = '../data/CRU.nc'
		if SSTfile == None:
			SSTfile = '../data/SST.nc'
		
		# extract coverage mask and temperature change for tas, masked
		self.cvgmasks = self.make_cvgmasks(CRUfile,SSTfile)
		self.dtas = self.make_dT_map(tasfile)
		self.dxax = self.make_dT_map(xaxfile)
		
		# make masked arrays for each period, where coverage >25 % of time
		self.dtasglob = np.ma.array(data=self.dtas, 
								   mask=np.zeros_like(self.dtas).astype('bool'))
		self.dtas1861 = np.ma.array(data=self.dtas,
						mask=~(self.cvgmasks['1861-1880'].mean(axis=0)>0.25))
		self.dtas2000 = np.ma.array(data=self.dtas,
						mask=~(self.cvgmasks['2000-2009'].mean(axis=0)>0.25))
		self.dxaxglob = np.ma.array(data=self.dxax, 
								   mask=np.zeros_like(self.dtas).astype('bool'))
		self.dxax1861 = np.ma.array(data=self.dxax,
						mask=~(self.cvgmasks['1861-1880'].mean(axis=0)>0.25))
		self.dxax2000 = np.ma.array(data=self.dxax,
						mask=~(self.cvgmasks['2000-2009'].mean(axis=0)>0.25))
		
		# grid longitudes and area weights
		self.glons,self.glats = np.meshgrid(np.arange(0,356,5),
											np.arange(-90,86,5))
		self.Awts = areaquad(self.glats,self.glons,self.glats+5,self.glons+5)

	
	def make_dT_map(self,ncfile,Tkey='tas'):
		""" Produce a global map of gridpoint change in temperature from
		1861--1880 to 2000--2009 
		"""
		
		ncf = nc.Dataset(ncfile,'r')
		out = {}
		for var in ncf.variables:
			out[str(var)] = ncf.variables[var][:]
		ncf.close()
		
		out['time'] = np.array([ dt.datetime.strptime(str(int(x)),'%Y%m%d') for 
								 x in out['time'] ])
		
		# calculate monthly maps and then use to take annual average
		yr = np.array([ x.year for x in out['time']])
		month = np.array([ x.month for x in out['time']])
		T = out[Tkey]
		
		# do month-by-month change in temperature calculation
		dT = np.array([ T[(yr>=2000)&(yr<=2009)&(month==x)].mean(axis=0) - \
						T[(yr>=1861) & (yr<=1880)&(month==x)].mean(axis=0) 
						for x in range(1,13) ])
		
		# take annual average accounting for mean month length
		months = np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])
		dT_yr = np.array([ dT[x] * months[x] / 365.25 for x in range(12) 
						   ]).sum(axis=0)
		
		return dT_yr
	
	def annual_mask(self,cvglnd,cvgsst,t,start=None,end=None):
		""" Given HadCRUT4 coverage, produce a month-by-month mask within
		start and end years
		"""
		
		# if value is above -500, treat as valid measurement
		lndmask0,sstmask0 = cvglnd.data > -500,cvgsst.data > -500
		
		# select period
		lndmask,sstmask = [ x[(t>=start) & (t<=end)] for x in [lndmask0,
															   sstmask0] ]
		
		# return mask
		return lndmask | sstmask
	
	def make_cvgmasks(self,CRUfile,SSTfile,starts=np.array([1861,2000]),
					  ends=np.array([1880,2009])):
		"""
		"""
		
		out = {}
		
		# load land and ocean Hadley data 
		ncf = nc.Dataset(CRUfile,'r')
		cvglnd = ncf.variables['temperature_anomaly'][:]
		ncf.close()
		
		ncf = nc.Dataset(SSTfile,'r')
		cvgsst = ncf.variables['sst'][:]
		t_cvgsst = ncf.variables['time'][:]
		ncf.close()
		
		# get year out of t_cvgsst
		t = np.array([ dt.datetime.strptime(str(int(t_cvgsst[x])),'%Y%m%d').year
					   for x in range(len(t_cvgsst)) ])
		
		# prepare dictionary
		for a in range(len(starts)):
			okey = '%s-%s' % (starts[a],ends[a])
			out[okey] = self.annual_mask(cvglnd,cvgsst,t,start=starts[a],
										 end=ends[a])
		
		# empty mask for full range period
		out['all'] = np.zeros_like(out[okey]).astype('bool')
		
		return out

#------------------------------------------------------------------------------#
# Functions to calculate TCR
#------------------------------------------------------------------------------#

def trendTCR(Fseries,Tseries,baseline=None,start=1970,end=2010,F2xCO2=3.44,
				  mode=None,use_index=False):
	""" Returns TCR calculated using the Otto et al. method.
	
	Fseries :: Forcing series, either as single 2 column array or as list [t,F]
	Tseries :: Temperature series, either as 2 columns array or as list [t,T]
	start :: first year for trend
	end :: last year for trend
	F2xCO2 :: Forcing associated with CO2 doubling, W m^-2
	"""
	
	if mode == 'single_array':
		t_F,F = Fseries[:,0].astype('int'),Fseries[:,1]
		t_T,T = Tseries[:,0].astype('int'),Tseries[:,1]
	else:
		t_F,F = Fseries[0].astype('int'),Fseries[1]
		t_T,T = Tseries[0].astype('int'),Tseries[1]
	
	if use_index == True:
		Fperiod = F[start:end+1]
		Tperiod = T[start:end+1]
	else:
		Fperiod = F[(t_F >= start) & (t_F <= end)]
		Tperiod = T[(t_T >= start) & (t_T <= end)]
	
	return F2xCO2 * s.linregress(Fperiod,Tperiod)[0]


def oneboxTCR(Fseries,Tseries,baseline=None,start=1861,end=2010,tau=4.0,
			   F2xCO2=3.44,mode=None,use_index=False):
	""" Returns just TCR using 1 box energy balance model.
	
	Fseries :: forcing series as 2-column array or [t,F]
	Tseries :: temperature series as 2-column array or [t,T]
	start :: first year of analysis (if use_index=False) or index of first
			 element in array (if use_index=True)
	end :: final year of analysis (if use_index=False) or index of final
		   element in array (if use_index=True)
	tau :: first time constant (default = 4 years)
	F2xCO2 :: radiative forcing for doubled CO2 (W m^-2)
	mode :: 'single_array' if Fseries,Tseries are 2-column arrays, otherwise
			something else.
	use_index :: defines method to select start and end points.
	
	"""
	
	if mode == 'single_array':
		t_F,F = Fseries[:,0].astype('int'),Fseries[:,1]
		t_T,T = Tseries[:,0].astype('int'),Tseries[:,1]
	else:
		t_F,F = Fseries[0].astype('int'),Fseries[1]
		t_T,T = Tseries[0].astype('int'),Tseries[1]
	
	# If use_index=true, then doing 1pctCO2 simulation so match the timing
	if use_index == True:
		Fperiod = F[start:end+1]
		Tperiod = T[start:end+1]
	else:
		Fperiod = F[(t_F >= start) & (t_F <= end)]
		Tperiod = T[(t_T >= start) & (t_T <= end)]
	
	# produce the decay function for convolution
	tdummy = np.arange(0,len(Fperiod))
	response = np.exp(-tdummy / tau)
	response = response / response.sum() # flip & normalise
	
	# lagged forcing
	laggedF = np.convolve(response,Fperiod)[:len(Fperiod)]
	
	# perform the curve fit to obtain TCR
	return F2xCO2 * s.linregress(laggedF,Tperiod)[0]


def diffTCR(Fseries,Tseries,baseline=[1860,1879],start=1970,end=2009,
			 F2xCO2=3.44,mode=None,use_index=False):
	""" Returns TCR calculated using the Otto et al. method.
	
	Fseries :: Forcing series, either as single 2 column array or as list [t,F]
	Tseries :: Temperature series, either as 2 columns array or as list [t,T]
	start :: Start year of latter analysis period
	end :: 	End year of latter analysis period
	period :: list of start and end years of period for comparison with baseline
	F2xCO2 :: Forcing associated with CO2 doubling (W m^-2)
	mode :: 'single_array' means Fseries, Tseries are a single array, if given
			as list [t,T],[t,F] then change to anything else.
	"""
	
	if mode == 'single_array':
		t_F,F = Fseries[:,0].astype('int'),Fseries[:,1]
		t_T,T = Tseries[:,0].astype('int'),Tseries[:,1]
	else:
		t_F,F = Fseries[0].astype('int'),Fseries[1]
		t_T,T = Tseries[0].astype('int'),Tseries[1]
	
	# if 1pctCO2 simulation, match timing
	if use_index == True: 
		t_T,t_F = np.arange(len(t_T)),np.arange(len(t_F))
		baseline = [t_T[baseline[0]],t_T[baseline[1]]]
		start = t_T[start]
		end = t_T[end]
	
	Fbase = F[(t_F >= baseline[0]) & (t_F <= baseline[1])].mean()
	Tbase = T[(t_T >= baseline[0]) & (t_T <= baseline[1])].mean()
	Fperiod = F[(t_F >= start) & (t_F <= end)].mean()
	Tperiod = T[(t_T >= start) & (t_T <= end)].mean()
	
	return F2xCO2 * (Tperiod - Tbase) / (Fperiod - Fbase)


def simpleTCR(Fseries=None,Tseries=None,baseline=[0,0],start=60,end=79,
			  use_index=True):
	""" Given 1pctCO2 annual temperatures, produces an estimate of the TCR
	"""
	
	t_T,T = Tseries[0],Tseries[1]
	
	if use_index == True:
		return T[start:end+1].mean() - T[baseline[0]:baseline[1]+1].mean()
	else:
		Tperiod = T[(t_T >= start) & (t_T <= end)]
		Tbaseline = T(t_T >= baseline[0]) & (t_T <= baseline[1])
		return Tperiod.mean() - Tbaseline.mean()

#------------------------------------------------------------------------------#
# Functions to output results
#------------------------------------------------------------------------------#

# F2xCO2 where randomly sampled, independent of main dF
F2xCO2_indep = np.random.randn(1000000) * (0.344/1.645) + 3.440

# calculate value at a chosen percentile
calc_pctile = lambda vals,pctile : np.interp(pctile*len(vals),
										   np.arange(len(vals))+1,np.sort(vals))
# report 5, 50, 95 percentiles
pds = lambda vals : np.array([ calc_pctile(vals,x) for x in [0.05,0.50,0.95] ])
# change in a time series over 1861--1880 to 2000--2009
delta61_09 = lambda t,val : val[(t>=2000)&(t<=2009)].mean() - val[(t>=1861)
															  &(t<=1880)].mean()

# function for linear regression, used in curve_fit with no intercept allowed
def linfunc(x,a):
	return a * x

def areaquad(lat1,lon1,lat2,lon2):
	""" Calculate a grid of area weighting coefficients, where each coefficient
	is the fraction of Earth's surface contained between the given lat-lons.
	"""
	
	lat1 = lat1 * np.pi / 180.
	lat2 = lat2 * np.pi / 180.
	
	return abs((np.sin(lat1) - np.sin(lat2)) * (lon1 - lon2) / 360. ) / 2.


def writeTCR(dat,scenario,calc,modRFname,obsRFname,baseline,start,end,models,
			 outfile,use_obs=True,use_index=False):
	""" Produces a text file listing the transient climate response calculated
	for different blending and masking methods.
	
	dat :: a mod.annual_blended_series instance
	calc :: TCR calculation method, e.g. mod.diffTCR
	modRFname :: name of forcing series to use for models
	obsRFname :: name of forcing series to use for observations
	baseline :: list of start and end dates of basetime or None
	start :: start of analysis period
	end :: end of analysis period
	models :: list of models to use
	outfile :: save location for output text file
	use_obs :: Boolean, stating whether to perform calculation on observational
		       temperature series in dat
	"""
	
	if '1pctCO2' in scenario:
		data = dat._1pctCO2
	elif 'hist_rcp85' in scenario:
		data = dat.hist_rcp85
	if use_obs == True:
		oFseries = [ dat.RF[obsRFname]['t'],dat.RF[obsRFname]['F'] ]
	
	models.sort()
	mFseries = [ dat.RF[modRFname]['t'],dat.RF[modRFname]['F'] ]
	mask = [ 'Txax','Txax','Had4' ] # global, global, Had4 coverage
	blend = [ 'T0','T','T' ] # air-only, blended, blended
	
	with open(outfile,'w') as f:
		f.write('Series\ttas_TCR\tblended_TCR\tHad4_TCR\n')
		
		# write observational estimate of TCR report same value in each column
		if use_obs == True:
			for obs in dat.Tobs.keys():
				Tseries = [ dat.Tobs[obs]['t'],dat.Tobs[obs]['T'] ]
				TCRs = [ calc(oFseries,Tseries,baseline,start,end,
						 use_index=use_index) for x in range(3) ]
				f.write( 'OBS_%s\t%s\n' % (obs,('\t').join('%.3f'%x for x 
										   in TCRs)) )
		
		# write TCR estimates for each model
		for model in models:
			Tserieslist = [ [ data[mask[x]][model]['t'],
							  data[mask[x]][model][blend[x]] ] 
							  for x in range(3) ]
			TCRs = [ calc(Fseries=mFseries,Tseries=Tseries,baseline=baseline,
						  start=start,end=end,use_index=use_index) for Tseries 
						  in Tserieslist ]
			f.write( '%s\t%s\n' % (model,('\t').join('%.3f'%x for x in TCRs)) )


def writeTCRmodelF(dat,scenario,calc,modRFname,obsRFname,baseline,start,
					end,models,outfile,use_obs=True,use_index=False):
	""" see function writeTCR, this only differs in that it uses model's own 
	forcing where available
	"""
	
	if '1pctCO2' in scenario:
		data = dat._1pctCO2
	elif 'hist_rcp85' in scenario:
		data = dat.hist_rcp85
	if use_obs == True:
		oFseries = [ dat.RF[obsRFname]['t'],dat.RF[obsRFname]['F'] ]
	
	models.sort()
	mFseries = [ dat.RF[modRFname]['t'],dat.RF[modRFname]['F'] ]
	mask = [ 'Txax','Txax','Had4' ] # global, global, Had4 coverage
	blend = [ 'T0','T','T' ] # air-only, blended, blended
	
	with open(outfile,'w') as f:
		f.write('Series\ttas_TCR\tblended_TCR\tHad4_TCR\n')
		
		# write observational estimate of TCR report same value in each column
		if use_obs == True:
			for obs in dat.Tobs.keys():
				Tseries = [ dat.Tobs[obs]['t'],dat.Tobs[obs]['T'] ]
				TCRs = [ calc(oFseries,Tseries,baseline,start,end,
						 use_index=use_index) for x in range(3) ]
				f.write( 'OBS_%s\t%s\n' % (obs,('\t').join('%.3f'%x for x 
										   in TCRs)) )
		
		# write TCR estimates for each model, if forcing for that model is not
		# available then use ensemble average
		for model in models:
			if model.split('_')[0] in dat.RF['allmodels']['models']:
				# column from which to extract this model's forcing
				col = model.split('_')[0]==dat.RF['allmodels']['models']
				
				modelFseries = [ dat.RF['allmodels']['t'],
								 dat.RF['allmodels']['F'][:,col] ]
				if len(modelFseries[1].shape) > 1:
					modelFseries[1] = np.ravel(modelFseries[1])
			else:
				modelFseries = mFseries
			
			Tserieslist = [ [ data[mask[x]][model]['t'],
							  data[mask[x]][model][blend[x]] ] 
							  for x in range(3) ]
			TCRs = [ calc(Fseries=modelFseries,Tseries=Tseries,baseline=baseline,
						  start=start,end=end,use_index=use_index) for Tseries 
						  in Tserieslist ]
			f.write( '%s\t%s\n' % (model,('\t').join('%.3f'%x for x in TCRs)) )


def write_decadal_PDFs(dat,outfile,T0s=None,DTs=None,F0s=None,DFs=None,
						  TCRfiles=None):
	""" Supplementary Table 1.
	
	This prints out the 5th, 50th and 95th percentiles of the TCR calculated
	given forcing and temperature changes and their errors, plus the
	inferred bias. Using Otto calculation with independent dF and dF2xCO2
	distributions.
	
	Calculation performed by sampling 1,000,000 times from each distribution.
	"""
	
	# For bias correction, use TCRs calculated for CMIP5 models using difference
	# calculation for each ending period.
	if TCRfiles == None:
		starts = np.array([1970,1980,1990,2000])
		ends = np.array([1979,1989,1999,2009])
		TCRfiles = [ '../data/TCR/diff_tsens/diffTCR_1861-1880_%i-%i.txt' % 
					(starts[x],ends[x]) for x in range(len(starts)) ]
	
	# If values aren't provided, then use Otto defaults
	if T0s == None:
		tT,T = dat.Tobs['HadCRUT4']['t'].astype('int'),dat.Tobs['HadCRUT4']['T']
		tF,F = dat.RF['otto']['t'].astype('int'),dat.RF['otto']['F']
		baseT = T[(tT>=1861) & (tT<=1880)].mean()
		baseF = F[(tF>=1861) & (tF<=1880)].mean()
		
		T0s = np.array([ T[(tT>=x)&(tT<=x+9)].mean() - baseT for x in [1970,
															  1980,1990,2000] ])
		F0s = np.array([ F[(tF>=x)&(tF<=x+9)].mean() - baseF for x in [1970,
															  1980,1990,2000] ])
		
		DTs = np.array([0.20,0.20,0.20,0.20])
		DFs = np.array([0.47,0.48,0.56,0.58])
		
	with open(outfile,'w') as f:
		# header, units of bias are tasTCR/blended-maskedTCR ratio
		f.write('Decade\tDeltaT\tError\tDeltaF\tError\tTCR_5pct\tTCR_50pct\t\
TCR95pct\tadjTCR_5pct\tadjTCR50pct\tadjTCR_95pct\tTCRbias_5pct\tTCRbias_50pct\
\tTCRbias95pct\n')
		
		# count through each file and get the TCRs, TCRbias
		for a in range(len(TCRfiles)):
			# TCR samples and calculated blending-masking bias as a fraction
			TCRs,TCRbias = fTCR_and_bias_PDF(T0=T0s[a],DT=DTs[a]/1.645,F0=F0s[a],
											DF=DFs[a]/1.645,skip_header=4,
				 			 				TCRfile=TCRfiles[a],tascol=1,
				 			 				hadcol=3)
			# bias corrected TCRs
			aTCRs = TCRs * TCRbias
			
			# Reference decade for label
			decade = TCRfiles[a].split('.')[-2].split('_')[-1]
			
			# Values at given percentiles for each
			pctiles = np.array([0.05,0.50,0.95])
			TCRdist = np.array([ calc_pctile(TCRs,x) for x in pctiles ])
			aTCRdist = np.array([ calc_pctile(aTCRs,x) for x in pctiles ])
			biasdist = np.array([ calc_pctile(TCRbias,x) for x in pctiles ])
			outdata = ('\t').join(np.array([ '%.2f' % x for x in 
									   np.hstack((TCRdist,aTCRdist,biasdist))]))
			
			# list of values used based on Otto et al.
			indata = ('\t').join(np.array([ '%.2f' % x for x in np.array([ 
											T0s[a],DTs[a],F0s[a],DFs[a] ]) ]))
			
			# write line
			f.write('%s\t%s\t%s\n' % (decade,indata,outdata) )


def EBM_performance_check(trueTCRfile,TCRfiles,EBMs,outfile,jointmodels,
						  trueTCR_skip_header=1,TCRfile_skip_header=4):
	""" Supplementary Table 2.
	 
	Calculates the performance of an energy-budget method (EBM), returning 
	median and mean biases, plus the standard deviation of the differences
	from canonical TCR.
	
	trueTCRfile :: path to IPCC_TCR file
	TCRfile :: path to calculated energy-budget TCR file
	EBMs :: list of energy budget model names
	outfile :: path to file in which to save results
	jointmodels :: list of models where forcing and canonical TCR available
	trueTCR_skip_header :: number of headear lines in trueTCRfile
	TCRfile_skip_header :: number of header plus obsTCR lines in TCRfiles
	"""
	
	# load true TCRs, skip 1 header
	with open(trueTCRfile,'r') as f:
		alltrueTCRs = { line.split('\t')[0] : float(line.strip().split('\t')[1]) 
					for line in f }
	
	# load TCRs calculated by method, skip 4 headers right for title +
	# 3 observational estimates
	def diffstats(TCRfile,alltrueTCRs,jointmodels):
		# open file, get list of models and TCRs
		with open(TCRfile,'r') as f:
			for skip in range(TCRfile_skip_header):
				dummy = next(f)
			
			modTCRs = { line.split('\t')[0] : np.array(
			line.strip().split('\t')[1:]).astype('float') for line in f }
		
		# extract pairings of TCRs
		trueTCRs = np.array([ alltrueTCRs[x.split('_')[0]] for x in 
							  modTCRs.keys() if x.split('_')[0] in
							  alltrueTCRs.keys() ])
		tasTCRs = np.array([ modTCRs[x][0] for x in modTCRs.keys() if 
							 x.split('_')[0] in alltrueTCRs.keys() ])
		hadTCRs = np.array([ modTCRs[x][2] for x in modTCRs.keys() if 
							 x.split('_')[0] in alltrueTCRs.keys() ])

		# now for models where both forcing and canonical is known
		joint_trueTCRs = np.array([ alltrueTCRs[x.split('_')[0]] for x in
									modTCRs.keys() if x.split('_')[0] in 
									jointmodels ])
		joint_tasTCRs = np.array([ modTCRs[x][0] for x in modTCRs.keys() if 
								   x.split('_')[0] in jointmodels ])
		joint_hadTCRs = np.array([ modTCRs[x][2] for x in modTCRs.keys() if 
								   x.split('_')[0] in jointmodels ])
		
		# return the ratios
		return np.array([ np.median(x) for x in [ trueTCRs/tasTCRs,
						  trueTCRs/hadTCRs, joint_trueTCRs/joint_tasTCRs,
						  joint_trueTCRs/joint_hadTCRs ] ])
	
	# call above function for each file and write to outfile
	with open(outfile,'w') as f:
		f.write('EBM_method\tall_tasfrac\tall_hadfrac\tjoint_tasfrac\tjoint_hadfrac\n')
		for a in range(len(TCRfiles)):
			outs = ['%.3f' % x for x in diffstats(TCRfiles[a],alltrueTCRs,
												  jointmodels) ]
			f.write('%s\t%s\n' % (EBMs[a],('\t').join(outs)))


def write_ensemble_pctile(TCRfiles,EBMs,outfile,obsTCRs=None,obsTCRs_row=None,
						  models=False,sims=False):
	""" Supplementary Table 3. Supplementary Table 4.
	 
	Writes a table containing either the mean TCR or the percentile at which
	the HadCRUT4 calculation falls for each energy-budget calculation and blend
	method
	
	TCRfiles :: list of TCR files
	EBMs :: list of energy-budget model names
	outfile :: save location
	obsTCR :: if None then use HadCRUT4, otherwise a list of TCRs for tas, xax,
			  had4 
	obsTCRs_row :: if obsTCR not provided and this is None, then this is a row
					number in the file referring to observed record to use.
	models :: List of models to use, if False then use all.
	sims :: e.g. ['r1i1p1'] means pick r1i1p1 sims only. Either this, or models
					
	"""
	
	pctile = lambda obsTCR, modTCR: 1.*(obsTCR > modTCR).sum(axis=0)/len(modTCR)
	
	with open(outfile,'w') as f:
		f.write('Method\ttas\txax\thad4\n')
		
		for a in range(len(EBMs)):
			TCRs = np.genfromtxt(TCRfiles[a],skip_header=1)[:,1:]
			if (obsTCRs_row == None) & (obsTCRs==None): # if none, use from file
				obsTCR = TCRs[1]
			elif obsTCRs_row != None:
				obsTCR = obsTCRs[obsTCRs_row]
			else:
				obsTCR = obsTCRs[a]
			
			modTCRs = TCRs[3:]
			
			# overwrite modTCRs if models are provided
			if (models != False) | (sims != False):
				with open(TCRfiles[a],'r') as f1:
					runs = np.array([ line.split('\t')[0] for line in f1])[4:]
				
				if models != False:
					use = np.array([ True if x.split('_')[0] in models else 
									 False for x in runs ])
				else:
					use = np.array([ True if x.split('_')[1] in sims else False
									 for x in runs ])
				
				modTCRs = modTCRs[use]
			
			pctiles = pctile(obsTCR,modTCRs)
			
			f.write('%s\t%s\n'%(EBMs[a],('\t').join('%.2f'%x for x in pctiles)))


def write_TCR_correction_stats(TCRfiles,EBMs,outfile,skip_header=1):
	""" Supplementary Table 5. 
	
	Produces a text file containing different methods of correction to
	blended or blended-masked TCR and compares their performance against air
	temperature-derived TCR by also reporting RMSE.
	
	TCRfiles :: list of files containing TCRs
	headers :: method(s) of calculating TCR, e.g. 'true','diff'
	outfile :: path to save location
	skip_header :: 1 if no observations in file, 4 if obs included
	"""
	
	blends = ['blended','blended-masked']
	
	# return line containing differences, standard deviations, fits and RMSE
	# of these corrections
	def return_TCR_correction_line(tasTCR,blendTCR):
		RMSE = lambda x,fitx: np.sqrt( ((x-fitx)**2).mean() )
		fitted = lambda x,y: x * s.linregress(x,y)[0] + s.linregress(x,y)[1]
		
		absdiff = (tasTCR - blendTCR).mean()
		absstd = (tasTCR - blendTCR).std(ddof=1)
		absRMSE = RMSE(tasTCR,blendTCR + absdiff)
		
		# changes made to linear fit
		ratio = curve_fit(linfunc,blendTCR,tasTCR)
		frac,fracstd = ratio[0][0],np.sqrt(ratio[1][0])
		fracRMSE = RMSE(tasTCR,blendTCR*frac)
		
		trend,intcpt = [ s.linregress(blendTCR,tasTCR)[x] for x in [0,1] ]
		fitRMSE = RMSE(tasTCR,fitted(blendTCR,tasTCR))
		
		return np.array([absdiff,absstd,absRMSE,frac,fracstd,
					     fracRMSE,trend,intcpt,fitRMSE])
	
	# write output to file
	with open(outfile,'w') as f:
		f.write('method_mask\tabsdiff\tabsstdev\tabsRMSE\tfracdiff\tfracstdev\t\
fracRMSE\ttrend\tintcpt\tfitRMSE\n')
		
		for a in range(len(EBMs)):
			TCRs = np.genfromtxt(TCRfiles[a],skip_header=skip_header)[:,1:]
			# get 2 lines, using blended (TCR column 1) and blended-masked
			# (column 2) versus tas-only (column 0)
			linesout = [ return_TCR_correction_line(TCRs[:,0],blendTCR) for
					 	 blendTCR in [TCRs[:,1],TCRs[:,2]] ]
			
			for b in range(2):
				f.write('%s_%s\t%s\n' % (EBMs[a],blends[b],
							   ('\t').join(['%.3f' % x for x in linesout[b]])) )


def write_diff_tsens_table(dat,baselines,starts,ends,models,outfile):
	""" Supplementary Table 6.
	
	Writes a table showing the calculated ensemble median TCR for different
	start and end periods using the difference method applied to the CMIP5
	historical simulations, and calculates the percentile at which the HadCRUT4
	historical series falls.
	
	baselines :: list of 2-element lists for start and end year of baseline
	starts :: list of start years of reference periods
	ends :: list of end years of reference periods
	outfile :: save location for output.
	"""
	
	pctile = lambda obsTCR, modTCR: 1.*(obsTCR > modTCR).sum(axis=0)/len(modTCR)
	
	for baseline,a in itertools.product(baselines,range(len(starts))):
			# first write output to a file for given time periods
			TCRfile = '../data/TCR/diff_tsens/diffTCR_%i-%i_%i-%i.txt' % \
						(baseline[0],baseline[1],starts[a],ends[a])
			writeTCRmodelF(dat=dat,scenario='hist_rcp85',calc=diffTCR,
					 modRFname='forster85',obsRFname='otto',use_obs=True,
					 baseline=baseline,start=starts[a],end=ends[a],
					 models=models,outfile=TCRfile,use_index=False)
	
	with open(outfile,'w') as f:
		f.write('start\tend\ttasTCR\tpctile\txaxTCR\tpctile\thad4TCR\tpctile\t\
				 HadCRUT4_TCR\n')
		
		for baseline,a in itertools.product(baselines,range(len(starts))):
			TCRfile = '../data/TCR/diff_tsens/diffTCR_%i-%i_%i-%i.txt' % \
						(baseline[0],baseline[1],starts[a],ends[a])
			
			period = TCRfile.split('_')[-1].replace('.txt','')
			base = TCRfile.split('_')[-2]
			
			TCRs = np.genfromtxt(TCRfile,skip_header=1)[:,1:]
			obsTCR = TCRs[1,0]
			modTCRs = TCRs[3:]
			
			# for each blending method report median and percentile at which 
			pctiles = pctile(obsTCR,modTCRs)
			medians = np.median(modTCRs,axis=0)
				
			f.write('%s\t%s\t%s\t%s\n' % (base,period,('\t').join( '%.2f'%x for 
						          x in np.hstack(zip(medians,pctiles))),obsTCR))




def make_temperature_ratio_table(dat,models,saveloc):
	""" Supplementary Table 7.
	
	Produce a list of CMIP5 models with the ratio of their temperature change
	from 1861--1880 to 2000--2009, taking global air temperature change and
	dividing it by blended temperature change.
	
	Use r1i1p1 simulation from each model.
	"""
	
	# ensure r1i1p1 simulation
	models = [ x for x in models if 'r1i1p1' in x ]
	models.sort()
	
	# define internal function to calculate temperature change ratio
	def calc_deltaT_ratio(modvals):
		t1 = (modvals['t']>=1861)&(modvals['t']<=1880)
		t2 = (modvals['t']>=2000)&(modvals['t']<=2009)
		
		dtas = modvals['T0'][t2].mean() - modvals['T0'][t1].mean()
		dxax = modvals['T'][t2].mean() - modvals['T'][t1].mean()
		
		return dtas / dxax
	
	ratios = np.array([ calc_deltaT_ratio(dat.hist_rcp85['Txax'][model]) for 
						model in models ])
	
	with open(saveloc,'w') as f:
		for a in range(len(ratios)):
			f.write('%s\t%.3f\n' % (models[a],ratios[a]))


def masked_zonal_sums(dT,saveloc='../data/results/zonal_mask_Tbias.txt'):
	""" Contributes to Supplementary Table 8
	
	Produces a text file showing the cumulative contribution to masking bias
	in tas-only from CMIP5 mean by latitude band.
	"""
	
	# calculate area-weighted hemispheric averages from available data for each
	# mask
	NH1861 = (dT.dtas1861 * dT.Awts)[18:].sum() /\
			 np.ma.array(dT.Awts,mask=dT.dtas1861.mask)[18:].sum()
	SH1861 = (dT.dtas1861 * dT.Awts)[:18].sum() /\
			 np.ma.array(dT.Awts,mask=dT.dtas1861.mask)[:18].sum()
	
	NH2000 = (dT.dtas2000 * dT.Awts)[18:].sum() /\
			 np.ma.array(dT.Awts,mask=dT.dtas2000.mask)[18:].sum()
	SH2000 = (dT.dtas2000 * dT.Awts)[:18].sum() /\
			 np.ma.array(dT.Awts,mask=dT.dtas2000.mask)[:18].sum()
	
	# as in hadCRUT4, replace missing data with hemispheric average
	f_dtas1861 = np.array(dT.dtas1861.data)
	f_dtas1861[18:][dT.dtas1861.mask[18:] == True] = NH1861
	f_dtas1861[:18][dT.dtas1861.mask[:18] == True] = SH1861
	
	f_dtas2000 = np.array(dT.dtas2000.data)
	f_dtas2000[18:][dT.dtas2000.mask[18:] == True] = NH2000
	f_dtas2000[:18][dT.dtas2000.mask[:18] == True] = SH2000
	
	# Now calculate the latitudinal weighted average
	dtas_Lwtd = (dT.dtasglob * dT.Awts).sum(axis=1)
	f_dtas1861_Lwtd = f_dtas1861.mean(axis=1) * dT.Awts.sum(axis=1)
	f_dtas2000_Lwtd = f_dtas2000.mean(axis=1) * dT.Awts.sum(axis=1)
	
	# Calculate cumulative contribution by latitude
	sumglobal = np.cumsum(dtas_Lwtd)
	sum1861 = np.cumsum(f_dtas1861_Lwtd)
	sum2000 = np.cumsum(f_dtas2000_Lwtd)
	
	data = np.vstack((np.arange(-87.5,90,5),sumglobal,sum2000,sum1861)).transpose()
	
	with open(saveloc,'w') as f:
		# write header
		f.write('Latitude\tGlobal\tmasked_2000-2009\tmasked_1861-1880\n')	
		# output data	
		for line in data:
			f.write('%.1f\t%.4f\t%.4f\t%.4f\n' % (line[0],line[1],line[2],
												  line[3]) )


def cmip_natvar(piC_dir,outfile):
	""" Supplementary Table 9 (column 3).
	
	If provided with the path to piControl tas temperature time
	series directory, produces a list of annual natural variability estimates
	for CMIP5 models based on 2 x standard deviation of detrended annual 
	variability
	
	piC_dir :: path to directory containing tas timeseries from piControl runs
	outfile :: path to file in which output will be saved
	models :: if None, 
	"""
	
	# list of files, if model list is provided then restrict to those
	piC_files = np.sort(os.listdir(piC_dir))
	
	with open(outfile,'w') as f:
		for piC_file in piC_files:
			# load data
			data = np.genfromtxt(piC_dir + piC_file)
			# annual smooth
			t = data[:,0]
			Tyr = data[:,1]
			# remove trend
			Tdetrend = Tyr - (t * s.linregress(t,Tyr)[0])
			# return annual variability as 2 x standard deviation
			Tvar = 2. * Tdetrend.std(ddof=1)
			# extract run name and write to file with variability
			model_run = piC_file.split('.')[0]
			f.write('%s\t%.2f\n' % (model_run,Tvar) )


def write_model_list(Hmodels,Tmodels,Hfileout,Tfileout):
	""" Supplementary Table 10.
	
	For historical-RCP8.5 simulation list (Hmodels), writes to Hfileout
	a list of models and simulations for each model that are used.
	For 1pctCO2 simulations (Tmodels) prints a list of models only to 
	Tfileout. Simulations not required as all are r1i1p1.
	"""
	
	models = list(set(Hmodel.split('_')[0] for Hmodel in Hmodels))
	models.sort()
	
	with open(Hfileout,'w') as f:
		for model in models:
			sims = np.array([ Hmodel.split('_')[1] for Hmodel in Hmodels if
							  Hmodel.split('_')[0] == model ])
			runs = np.array([ int(sim.split('i')[0][1:]) for sim in sims ])
			sims = sims[ np.argsort(runs) ]
			f.write('%s\t%s\n' % (model,(', ').join(sims)))
	
	with open(Tfileout,'w') as f:
		for Tmodel in Tmodels:
			f.write(Tmodel.split('_')[0]+'\n')


def final_TCR_ranges(dat,correction,TCRfile,saveloc,DT=0.20,F2x0=3.44,DF2x=0.344,
					 sampleN=1e6):
	""" Supplementary Table 12. Supplementary Table 13.
	
	dat :: dat class from module
	correction :: 'trend' or 'sample'
	saveloc :: output save file location
	DT :: assumed 5--95 % error in temperature change
	F2x0 :: doubled CO2 forcing 
	DF2x :: 5--95 % error in doubled CO2 forcing
	"""
	
	# Get dT distributions
	T0 = delta61_09( dat.Tobs['HadCRUT4']['t'],dat.Tobs['HadCRUT4']['T'])
	
	# Get inputs for forcing from LC15 code output
	F0 = delta61_09(dat.RF['otto']['t'],dat.RF['otto']['F'])
	dFin = np.genfromtxt('../data/forcing/dF.dat')[:sampleN]
	dF2xin = np.genfromtxt('../data/forcing/dF2xCO2.dat')[:sampleN]
	
	# generate scale factors and temperature changes to use
	DFscalefactors = np.append( np.arange(0.25,2.1,0.25), np.ones(8) )
	DTs = np.append( np.ones(8) * 0.2, np.arange(0.05,0.41,0.05) )
	
	# prepare correction
	# generate alpha correction factor using modelled TCR values
	modTCRs = np.genfromtxt(TCRfile,skip_header=4)[:,1:]
	
	if correction == 'trend':
		fit = curve_fit(linfunc,modTCRs[:,2],modTCRs[:,0])
		alpha = np.random.randn(sampleN) * np.sqrt(fit[1][0]) + fit[0][0]
	if correction == 'sample':
		alpha = np.random.choice(modTCRs[:,0] / modTCRs[:,2],sampleN)
	
	# open output file to write to
	with open(saveloc,'w') as f:
		# write header
		f.write('Tchange\tTerror\tF05\tF50\tF95\tbmTCR05\tbmTCR50\tbmTCR95\ttasTCR05\ttasTCR50\ttasTCR95\n')
		
		# go through each set of forcing and temperature uncertainties
		for a in range(len(DTs)):
			# generate dT for given uncertainty
			dT = np.random.randn(sampleN) * DTs[a]/1.645 + T0
			# calculate dF and TCR PDFs
			dF,TCR = internal_dF_TCR_calc(dT,dFin,dF2xin,F0,F2x0=3.44,DF2x=0.344,
										  DFscalefactor=DFscalefactors[a])
			# apply correction
			aTCR = alpha * TCR
			
			# calculate ranges
			ranges = np.hstack(( pds(dF), pds(TCR), pds(aTCR) ))
			# build output line
			line = '%.2f\t%.2f\t%s\t\n' % (T0,DTs[0], ('\t').join(['%.3f' % x 
															for x in ranges])  )
			# write output line
			f.write(line)


def internal_dF_TCR_calc(dT,dFin,dF2xin,F0,DFscalefactor,F2x0,DF2x):
	""" Called by final_TCR_ranges()
	Returns samples of forcing PDF and calculated TCR PDF
	
	dT :: samples of temperature change in K
	dFin :: samples of forcing from LC15 code output
	dF2xin :: samples of 2xCO2 forcing from LC15 code output
	F0 :: median forcing value to use
	DFscalefactor :: multiplicative scale factor to apply to 5--95% LC15 forcing
					 range
	F2x0 :: average forcing for 2xCO2
	DF2x :: 5--95 % error in forcing for 2xCO2 (assuming Gaussian)
	
	We use LC15 code output to preserve correlation between dF and dF2xCO2.
	"""
	
	# scale whole LC forcing distribution to match provided median
	dFcentred = dFin * F0 / np.median(dFin)
	# scale distribution either side of median
	dF = np.median(dFcentred) + ( dFcentred - np.median(dFcentred) ) * \
		 (DFscalefactor)
	# ensure no zeros
	dF[dF==0] = 1.e-23
	
	# scale dF2xCO2
	dF2xcentred = dF2xin * F2x0 / dF2xin.mean()
	# scale centred values to make spread match:
	dF2x = dF2xcentred.mean() + (dF2xcentred - dF2xcentred.mean())  * (DF2x/
		   1.645) / dF2xcentred.std(ddof=1)
	
	# return TCR value
	return dF,dF2x * dT / dF


def writeTCR_tsensitivity(dat,mask,RF,models,modelF=False):
	""" Produces data for Supplementary Figure 3.
	
	data :: dat.hist_rcp85[mask] dictionary
	RF :: dat.RF[RFseries] dictionary
	models :: list of models to use
	"""
	
	data = dat.hist_rcp85[mask]
	
	# internal function to write a single text file given a specific EBM method
	# and temperature reconstruction combo
	def write_func(data,RF,models,blend,calc,baseline,starts,ends,outfile,
				   modelF=False):
		# prep values for calculations
		models.sort()
		
		# write output to file
		with open(outfile,'w') as f:
			f.write('Model\t%s\n' % ( ('\t').join([ '%i'%end for end in ends])))
			
			for model in models:
				Tseries = [ data[model]['t'].astype('int'),data[model][blend] ]
				
				# derive Fseries
				mod1 = model.split('_')[0]
				Fmodellist = dat.RF['allmodels']['models']
				
				if (modelF == False) | (mod1 not in Fmodellist):
					Fseries = [ RF['t'],RF['F'] ]
				else:
					Fseries = [ dat.RF['allmodels']['t'], 
					np.ravel(dat.RF['allmodels']['F'][:,mod1 == Fmodellist]) ]
				
				# prepare and write output
				TCRs = [ calc(Fseries=Fseries,Tseries=Tseries,baseline=baseline,
						 start=starts[x],end=ends[x]) for x 
						 in range(len(starts)) ]
				f.write('%s\t%s\n' %(model,('\t').join('%.2f'%x for x in TCRs)))
	
	# prepare default values
	blends = [ 'T0', 'T' ]
	blendnames = [ 'tas','xax' ]
	calcs = [ diffTCR, oneboxTCR, trendTCR ]
	methods = [ 'diff', 'onebox', 'trend']
	allstarts = [ np.arange(1970,2090), np.ones(121)*1861, np.ones(121)*1970 ]
	allends = [ np.arange(1979,2099),np.arange(1979,2100),np.arange(1979,2100) ]
	
	# count through all combinations of methods and temperature reconstruction
	# and write output
	for a,b in itertools.product(range(len(blends)), range(len(calcs))):
		outfile = '../data/TCR/tsensitivity/histrcp85_%s_%s.txt' % (methods[b],
															      blendnames[a])
		
		write_func(data,RF,models=models,blend=blends[a],
								  calc=calcs[b],baseline=[1861,1880],
								  starts=allstarts[b],ends=allends[b],
								  outfile=outfile,modelF=modelF)


def fTCR_and_bias_PDF(T0=0.75,DT=0.20/1.645,F0=1.95,DF=0.58/1.645,skip_header=4,
		 			 TCRfile='../data/TCR/all/allhist_diffTCR.txt',tascol=1,
					 hadcol=3):
	""" Called by code for Supplementary Table 1, Supplementary Figure 1.
	
	Returns 1,000,000 length arrays containing calculated TCRs plus  
	random sampled bias adjustments based on CMIP5 biases. Assumes Gaussian
	uncertainties. 
	
	TCR calculations assume Gaussian distribution.
	
	T0 = temperature change
	DT = variance in temperature change estimate
	F0 = forcing change
	DF = variance in forcing change estimate
	"""
	
	# 100,000 TCR values from Gaussian distribution of Ts and Fs
	Ts = np.random.randn(1000000) * DT + T0
	Fs = np.random.randn(1000000) * DF + F0
	TCRs = F2xCO2_indep * Ts/Fs
	
	# calculate bias adjustment
	data = np.genfromtxt(TCRfile,skip_header=skip_header)
	TCRdiffs = data[:,tascol] / data[:,hadcol]
	TCRbias = np.random.choice(TCRdiffs,1000000)
	
	return TCRs,TCRbias

#------------------------------------------------------------------------------#
# Functions to generate figures for main paper
#------------------------------------------------------------------------------#

def plot_scenario_mediandiff(dat,models,labels=['blended','blended-masked'],
							 saveloc=None):
	""" Figure 1. 
	
	Plots 2 panes: left is median timeseries of temperature from each blending 
	method and right is the difference between median of other blending methods 
	versus tas-only.
	"""
	
	# prepare cmip dictionary containing median temperature series
	data = dat.hist_rcp85
	cmip = { 'Txax' : { 'T0' : 1,'T' : 1}, 'Had4' : { 'T0' : 1,'T' :1} }
	
	for mask in cmip.keys():
		for blend in cmip[mask].keys():
			ensembleT = np.array([data[mask][model][blend] for model in models])
			cmip[mask][blend] = np.median(ensembleT,axis=0)
	
	# labelled data
    
	t = data['Had4'][models[0]]['t'].astype('int')
	tas,xax,had4 = cmip['Txax']['T0'],cmip['Txax']['T'],cmip['Had4']['T']
	baseline = lambda t,T,start,end : T - T[(t>=start)&(t<=end)].mean()
	tas,xax,had4 = [ baseline(t,x[:len(t)],1861,1880) for x in [tas,xax,had4] ]
	
	# prepare figure and axes
	fig = plt.figure(figsize=(8,4))
	ax1 = fig.add_subplot(1,2,1,ylim=[-0.4,1.2])
	ax2 = fig.add_subplot(1,2,2)
	
	# set xlabels and xticks
	for ax in [ax1,ax2]:
		ax.set_xticks(np.arange(1880,2021,40))
		ax.set_xlabel('Year',fontsize=16)
	
	ax1.set_ylabel('$\Delta$T ($^\circ$C)',fontsize=16,labelpad=-5)
	ax2.set_ylabel('$\Delta$T - $\Delta$T$_{tas}$ ($^\circ$C)',fontsize=16,
				   labelpad=0)
	
	ax1.plot(t,tas,'r-',label='tas-only CMIP5')	
	ax1.plot(t,xax,'mo-',label='blended CMIP5',markersize=3.5)
	ax1.plot(t,had4,'b^-',label='blended-masked CMIP5',markersize=2.3)
	
	ax2.plot(t,xax-tas,'mo-',markersize=3)
	ax2.plot(t,had4-tas,'b^-',markersize=2.5)
	
	# add observations
	obst = dat.Tobs['HadCRUT4']['t'].astype('int')
	obst,obsT = obst[obst>=1861],dat.Tobs['HadCRUT4']['T'][obst>=1861]
	ax1.plot(obst,obsT,'k-',linewidth=2.5,alpha=0.5,label='HadCRUT4 observed')
	
	leg = ax1.legend(loc='upper center',frameon=False,handletextpad=0)
	
	for ltext in leg.get_texts():
		ltext.set_fontsize(12)
	for ax in fig.get_axes():
		for ylab in ax.get_yticklabels():
			ylab.set_fontsize(11)
		for xlab in ax.get_xticklabels():
			xlab.set_fontsize(11)
	
	# labelling of subplots
	ax1.text(1865,1.07,'(a)',fontsize=20)
	ax2.text(1990,0.025,'(b)',fontsize=20)
	
	fig.subplots_adjust(0.08,0.12,0.98,0.97,wspace=0.26)
	
	if saveloc != None:
		fig.savefig(saveloc)
    

def kate(dat,models):
        # prepare cmip dictionary containing median temperature series
    data = dat.hist_rcp85
    cmip = { 'Txax' : { 'T0' : 1,'T' : 1}, 'Had4' : { 'T0' : 1,'T' :1} }
    
    for mask in cmip.keys():
        for blend in cmip[mask].keys():
            ensembleT = np.array([data[mask][model][blend] for model in models])
            cmip[mask][blend] = np.median(ensembleT,axis=0)
    
    # labelled data
    
    t = data['Had4'][models[0]]['t'].astype('int')
    tas,xax,had4 = cmip['Txax']['T0'],cmip['Txax']['T'],cmip['Had4']['T']
    baseline = lambda t,T,start,end : T - T[(t>=start)&(t<=end)].mean()
    tas,xax,had4 = [ baseline(t,x[:len(t)],1861,1880) for x in [tas,xax,had4] ]
    return tas,xax,had4

    
def TCR_histograms_main(saveloc=None):
	""" Figure 2.
	
	Produces histograms of the model-calculated TCRs for blended-masked and
	tas-only simulations, plotting the HadCRUT4 value on each one.
	"""
	
	TCRs = np.genfromtxt('../data/TCR/all/allhist_diffTCR.txt',
						 skip_header=1)[:,1:] 
	
	fig = plt.figure()
	
	ax1 = fig.add_subplot(1,2,1,xlim=[0,4],ylim=[0,28])
	ax2 = fig.add_subplot(1,2,2,xlim=[0,4],ylim=[0,28])
	
	# add axis labels, tick properties
	for ax in [ax1,ax2]:
		ax.set_xlabel('TCR ($^\circ$C)',fontsize=20)
		
		for xtl in ax.get_xticklabels():
			xtl.set_fontsize(15)
		for ytl in ax.get_yticklabels():
			ytl.set_fontsize(15)
	
	ax1.set_ylabel('N',fontsize=20)
	
	# add panel labels
	ax1.text(0.1,25,'(a)',fontsize=25)
	ax2.text(0.1,25,'(b)',fontsize=25)
	
	# plot a line showing the HadCRUT4 observational estimate
	obsTCR,modTCR = TCRs[1,0],TCRs[3:]
			
	# prepare histograms
	pdf1 = ax1.hist(modTCR[:,2],bins=10,histtype='stepfilled',alpha=0.3,
					color='blue')
	pdf2 = ax2.hist(modTCR[:,0],bins=10,histtype='stepfilled',alpha=0.3,
					color='red')
	
	plt.setp(pdf1[2],label='CMIP5 blended-\nmasked TCR')
	plt.setp(pdf2[2],label='CMIP5 tas-only TCR')
	
	# plot obs-based TCR
	pct1 = 100. * (obsTCR > modTCR[:,2]).sum() / len(modTCR[:,0])
	pct2 = 100. * (obsTCR > modTCR[:,0]).sum() / len(modTCR[:,2])
	
	label1 = 'HadCRUT4 blended-\nmasked TCR'
	label2 = 'HadCRUT4 blended-\nmasked TCR'
	
	ax1.plot([obsTCR,obsTCR],ax1.get_ylim(),'k-',label=label1,linewidth=2)
	ax2.plot([obsTCR,obsTCR],ax2.get_ylim(),'k-',label=label2,linewidth=2)
	
	# label each and add to legend
	
	leg1 = ax1.legend(loc='upper right',frameon=False,handletextpad=0.5)
	leg2 = ax2.legend(loc='upper right',frameon=False,handletextpad=0.5)
	
	for leg in [leg1,leg2]:
		for ltext in leg.get_texts():
			ltext.set_fontsize(15)
	
	fig.set_size_inches(10.5,5.5)
	fig.subplots_adjust(0.07,0.11,0.95,0.98,wspace=0.12)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_TCR_relationship(TCRfiles,skip_header=1,axlims=[1,2.5],saveloc=None,
						  headers='',figsize=(4,4),dpi=None,writefit=False):
	""" Figure 3, Supplementary Figure 2, Supplementary Figure 5.
	 
	Plots TCR derived for blended or blended-masked as a function of the TCR
	derived for air temperature. Used for Figure 1 and Figure 3.
	
	TCRfiles :: one file path or string of paths
	skip_header :: if no obs lines in file = 1, otherwise 4
	axlims :: range of TCRs to encompass
	headers :: list of subplot headers 
	"""
	
	if type(TCRfiles) is str:
		TCRfiles = [TCRfiles]
	if type(headers) is str:
		headers = [headers]

	# work out the shape in which to arrange the subplots 
	Nxs = [ 0,1,1,1,2,2,2 ]
	Nys = [ 0,1,2,3,2,3,3 ]
	Nx,Ny = Nxs[len(headers)],Nys[len(headers)]

	labels = ['blended','blended-masked']
	markers = ['mo','b^']
	lines = ['m-','b--']

	fig = plt.figure(figsize=figsize)

	for a in range(len(headers)):
		TCRfile,header = TCRfiles[a],headers[a]
		TCRs = np.genfromtxt(TCRfile,skip_header=skip_header)[:,1:]
		ax = fig.add_subplot(Nx,Ny,a+1,xlim=axlims,ylim=axlims,title=header)
		
		# axis labels
		ax.set_xlabel('tas-only TCR ($^\circ$C)',fontsize=20)
		ax.set_ylabel('Inferred TCR ($^\circ$C)',fontsize=20)
		
		# 1:1 line
		x = np.array(ax.get_xlim())
		ax.plot(axlims,axlims,'k:')
												
		# add TCRs and fits
		for b in range(len(labels)):			
			ax.plot(TCRs[:,0],TCRs[:,b+1],markers[b],label=labels[b],alpha=0.5)
			#fit = s.linregress(TCRs[:,0],TCRs[:,b+1])
			fit = curve_fit(linfunc,TCRs[:,0],TCRs[:,b+1])[0]
			#ax.plot(x,x*fit[0]+fit[1],lines[b],label='  ')
			if writefit == False:
				label = '  '
			else:
				label = r'$\alpha$ = %.3f' % curve_fit(linfunc,TCRs[:,b+1],
													   TCRs[:,0])[0][0]
			ax.plot(x,x*fit[0],lines[b],label=label)
		
		if a == 0:
			leg = plt.legend(loc='upper left',frameon=False,handletextpad=0,
							 ncol=(2 if len(headers)==1 else 1),
							 numpoints=1,columnspacing=0)
			for ltext in leg.get_texts():
				ltext.set_fontsize(15)

	if len(headers) == 1:
		fig.subplots_adjust(0.12,0.12,0.95,0.95)
	else:
		fig.tight_layout()
	
	if saveloc != None:
		if '.png' in saveloc:
			fig.savefig(saveloc,dpi=dpi)
		else:
			fig.savefig(saveloc)


def plot_ranges(saveloc='../figures/Figure4.pdf'):
	""" Figure 4.
	 
	Horizontal bar plot of 5--95% ranges of CMIP5 and energy-budget
	inferred TCR along with MMM or best estimate.
	"""
	
	yvals = [0.5,1,1.5,2.0,2.5]
	starts = [0.9,0.85,0.81,1.05,1.2] # lower limits
	widths = [ 1.1,1.79,1.32,2.21,1.2] # full range size
	bests = [ 1.33,1.34,1.41,1.65,1.8] # best estimates
	labels = ['Otto observed\nblended-masked\nenergy-budget',
			  'Otto updated $\Delta F$\nblended-masked\nenergy-budget',
			  'CMIP5\nblended-masked\nrange',
			  'Observation-based\ninferred tas-only\nenergy-budget',
			  'CMIP5 tas-only\nrange' ]
	xshift = [0.8,0.8,0.8,0.8,0.8] # text shift 
	yshift = [0.05,0.05,0.05,0.05,0.075] # text shift
	
	# prepare figure
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1,xlim=[0,3.6],ylim=[0.3,3.3],yticklabels=[],
						 yticks=[0.65,1.15,1.65])
	ax.set_xlabel('Transient Climate Response ($^\circ$C)',fontsize=20)
	
	for xtl in ax.get_xticklabels():
		xtl.set_fontsize(16)
	
	for a in range(len(yvals)):
		if 'blended-masked' in labels[a]:
			bar = ax.barh(bottom=yvals[a], width=widths[a], left=starts[a],
						  height=0.3,ecolor='black',color='blue',alpha=0.5)
		if 'tas-only' in labels[a]:
			tasbar = ax.barh(bottom=yvals[a], width=widths[a], left=starts[a],
						  	 height=0.3,alpha=0.5,color='red')
		line = ax.plot([bests[a],bests[a]],[yvals[a],yvals[a]+0.3],linewidth=3,
				 color='black')
		ax.text(starts[a]-xshift[a],yvals[a]+yshift[a],labels[a])
	
	plt.setp(line,label='Best estimate')
	plt.setp(bar,label='$5-95 \%$ blended-masked range')
	plt.setp(tasbar,label='$5-95 \%$ tas-only range')
	
	leg = ax.legend(frameon=False,loc='upper left',ncol=2)
	
	fig.subplots_adjust(0.05,0.1,0.95,0.95)
	
	fig.savefig(saveloc)


#------------------------------------------------------------------------------#
# Functions to generate supplementary figures
#------------------------------------------------------------------------------#

def plot_decadal_TCR_PDFs(T0s=None,DTs=None,F0s=None,DFs=None,TCRfiles=None,
						  saveloc=None):
	""" Supplementary Figure 1.
	
	Plot PDFs of calculated TCR using HadCRUT4 and bias corrected to 
	tas-only for each possible reference decade. 
	"""
	
	if TCRfiles == None:
		starts = np.array([1970,1980,1990,2000])
		ends = np.array([1979,1989,1999,2009])
		TCRfiles = [ '../data/TCR/diff_tsens/diffTCR_1861-1880_%i-%i.txt' % 
					(starts[x],ends[x]) for x in range(len(starts)) ]
	
	# If values aren't provided, then use Otto defaults
	if T0s == None:
		T0s = np.array([0.22,0.39,0.57,0.75])
		DTs = np.array([0.20,0.20,0.20,0.20])
		F0s = np.array([0.75,0.97,1.21,1.95])
		DFs = np.array([0.47,0.48,0.56,0.58])
	
	fig = plt.figure()
	
	for a in range(len(TCRfiles)):
		# get distributions fo TCR and fractional bias correction
		TCRs,TCRbias = fTCR_and_bias_PDF(T0=T0s[a],DT=DTs[a]/1.645,F0=F0s[a],
										DF=DFs[a]/1.645,skip_header=4,
			 			 				TCRfile=TCRfiles[a],tascol=1,
			 			 				hadcol=3)
		aTCRs = TCRs * TCRbias
		
		# get title
		decade = TCRfiles[a].split('.')[-2].split('_')[-1]
		
		# prepare axis
		ax = fig.add_subplot(2,2,a+1,title=decade)
		
		# plot histograms
		h1 = ax.hist(TCRs,bins=np.linspace(0,4,150),histtype='step',normed=True,
					 label='HadCRUT4 TCR')[2]
		h2 = ax.hist(aTCRs,bins=np.linspace(0,4,150),histtype='step',
					 normed=True,label='tas-only TCR')[2]
		
		if a == 0:
			leg = ax.legend(loc='upper right',frameon=False)
		
		if a % 2 == 0:
			ax.set_ylabel('P density')
		if a > 1:
			ax.set_xlabel('TCR ($^\circ$C)')
	
	fig.set_size_inches(10,8)
	fig.subplots_adjust(0.08,0.08,0.96,0.95)
	
	if saveloc != None:
		fig.savefig(saveloc)


def TCR_histograms(saveloc=None):
	""" Supplementary Figure 2.
	
	Produces histograms of the model-calculated TCRs for each of the methods
	and also adds a line for the HadCRUT4-calculated value.
	"""
	
	TCRs = np.array([ np.genfromtxt('../data/TCR/all/allhist_%sTCR.txt' % x,
					  skip_header=1)[:,1:] for x in ['diff','onebox','trend'] ])
	titles = ['Difference','One-box','Trend']
	
	fig = plt.figure()
	for a in range(0,3):
		ax1 = fig.add_subplot(2,3,a+1,xlim=[0,4],ylim=[0,35],
							  title=titles[a]+' CMIP5 tas-only',ylabel='N')
		ax2 = fig.add_subplot(2,3,a+4,xlim=[0,4],ylim=[0,35],ylabel='N',
							  xlabel='TCR ($^\circ$C)',
							  title=titles[a]+' CMIP5 blended-masked')
		
		# plot a line showing the HadCRUT4 observational estimate
		obsTCR,modTCR = TCRs[a][1,0],TCRs[a][3:]
				
		# prepare histograms
		ax1.hist(modTCR[:,0],bins=10,histtype='stepfilled',color='red',
				 alpha=0.3)
		ax2.hist(modTCR[:,2],bins=10,histtype='stepfilled',color='blue',
				 alpha=0.3)
		
		ax1.plot([obsTCR,obsTCR],ax1.get_ylim(),'k-',label='HadCRUT4 observed',
				  linewidth=2)
		ax2.plot([obsTCR,obsTCR],ax2.get_ylim(),'k-',label='HadCRUT4 observed',
				  linewidth=2)
		
		leg = ax1.legend(loc='upper center')
		for ltext in leg.get_texts():
			ltext.set_fontsize(9)
	
	fig.set_size_inches(12,8)
	fig.subplots_adjust(0.06,0.08,0.95,0.95)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_tsensitivities(datadir,files,titles,nx,ny,xlim,ylim,saveloc=None):
	""" Supplementary Figure 4. 
	
	datadir :: path to directory containing input files
	files :: list of tsensitivity TCR files to use
	titles :: subplot titles
	nx :: number of columns of subplots
	ny :: number of rows of subplots
	saveloc :: where to save figure
	"""
			
	fig = plt.figure(figsize=(12,8))
	
	for a in range(len(files)):
		ax = fig.add_subplot(ny,nx,a+1,title=titles[a],xlim=xlim,ylim=ylim)
		title = ax.title.set_fontsize(12.5)
		
		data = np.genfromtxt(datadir+files[a])[:,1:]
		
		t = data[0]
		T = data[1:]
		
		ax.plot(t,T.transpose(),'k-',alpha=0.3,linewidth=0.5)
		ax.plot(t,np.median(T,axis=0),'k-',linewidth=1.5)
			
		if a < (ny-1)*nx:
			ax.set_xticklabels([])
		if a%nx != 0:
			ax.set_yticklabels([])
		if a%nx == 0:
			ax.set_ylabel('TCR ($^\circ$C)')
		if a >= (ny-1) * nx:
			ax.set_xlabel('End Year of Analysis')
	
	fig.subplots_adjust(0.1,0.1,0.95,0.95)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_CMIP5_v_obs(dat,models,saveloc=None):
	""" Supplementary Figure 5.
	
	Plots the CMIP5 median and 5--95% ranges for the global air temperature
	and masked-blended HadCRUT4 equivalent along with the observed HadCRUT4
	time series.
	"""
	
	# prepare cmip dictionary containing median temperature series
	data = dat.hist_rcp85
	cmip,cmip05,cmip95 = ( { 'Txax' : { 'T0' : 1,'T' : 1}, 
							'Had4' : { 'T0' : 1,'T' :1} } for x in range(3) )
	
	baseline = lambda t,T,start,end : T - T[(t>=start)&(t<=end)].mean()
	
	# baseline each time series
	for mask,model,blend in itertools.product(cmip.keys(),models,
											  cmip[cmip.keys()[0]].keys()):
		data[mask][model][blend] = baseline(data[mask][model]['t'],
											data[mask][model][blend],
											1861,1880)
	
	# get time series
	t = data['Had4'][model]['t']
	# generate dictionaries of 5th, 50th and 95th percentile for graphing
	for mask in cmip.keys():
		for blend in cmip[mask].keys():
			ensembleT = np.array([data[mask][model][blend] for model in models])
			# basline
			cmip05[mask][blend] = np.array([ calc_pctile(ensembleT[:,x],0.05) 
								for x in np.arange(len(t)) ])
			cmip[mask][blend] = np.median(ensembleT,axis=0)
			cmip95[mask][blend] = np.array([ calc_pctile(ensembleT[:,x],0.95) 
								for x in np.arange(len(t)) ])
	
	# prepare figure
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1,xlim=[1861,2020])
	
	ax.set_xlabel('Year',fontsize=16)
	ax.set_ylabel('Temperature anomaly ($^\circ$C)',fontsize=16)
	
	# air temperature models
	ax.plot(t,cmip['Txax']['T0'][:len(t)],'r-',linewidth=1.5,alpha=0.7,
			label='CMIP5 tas-only')
	ax.fill_between(t,cmip05['Txax']['T0'][:len(t)],
					cmip95['Txax']['T0'][:len(t)],alpha=0.3,color='red')
	
	# models consistent with HadCRUT4
	ax.plot(t,cmip['Had4']['T'],'b-',linewidth=1.5,alpha=0.7,
			label='CMIP5 blended-masked')
	ax.fill_between(t,cmip05['Had4']['T'],cmip95['Had4']['T'],alpha=0.3,
					color='blue')
	
	# obs
	ax.plot(dat.Tobs['HadCRUT4']['t'],dat.Tobs['HadCRUT4']['T'],'k-',
			label='HadCRUT4 annual',linewidth=2)
	
	# format and save figure
	leg = plt.legend(loc='upper left',frameon=False)
	
	fig.set_size_inches(10,6)
	fig.subplots_adjust(0.1,0.1,0.95,0.95)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_coverage_Tmaps(dtass,saveloc='../figures/SI_Figure6.pdf'):
	""" Supplementary Figure 6.
	
	Plots 3 maps showing the temperature change over the provided dtas 
	period for each of the provided dtas fields (which should be masked to match
	a given period's coverage.
	"""
	
	titles = [ 'Global','2000-2009 coverage','1861-1880 coverage']
	dt
	fig = plt.figure()
	
	for a in range(3):
		ax = fig.add_subplot(1,3,a+1,title=titles[a])
		m = Basemap(projection='robin',resolution='c',lon_0=0,ax=ax)
		m.drawcoastlines()
		
		# flip the x,y and 
		if a == 0:
			glons,glats = np.meshgrid(np.arange(-177.5,180,5),np.arange(-87.5,90,5))
			x,y = m(glons,glats)
		
		# reorganise dtas values for -180 to 180 deg instead of 0 to 360 deg
		dtas0 = np.ma.array(np.hstack((dtass[a][:,36:],dtass[a][:,:36])),
						mask=np.hstack((dtass[a][:,36:].mask,dtass[a][:,:36].mask)))
		
		img = m.contourf(x,y,dtas0,levels=np.linspace(-3.7,3.7,21),
						 cmap='seismic')
		
	fig.subplots_adjust(0.01,0.1,0.99,0.95,wspace=0.01)
	fig.set_size_inches(12,3.3)
	
	cax = fig.add_axes((0.3,0.07,0.4,0.03),
					   title='$\Delta$ T$_{1861-2009}$ ($^\circ$C)')
	cbar = fig.colorbar(img,cax=cax,orientation='horizontal',
						ticks=np.linspace(-3,3,7))
	
	fig.savefig(saveloc)


def zonal_dT_profiles(dT,saveloc=None):
	""" Supplementary Figure 7.
	
	Plots zonal profiles of temperature change from 1861--1880 to 2000--2009,
	on the left the absolute change and on the right the area-weighted change.
	"""
	
	# define variables to use
	lats = np.arange(-87.5,90,5)
	dtas_L = dT.dtasglob.mean(axis=1).data
	dtas1861_L = dT.dtas1861.mean(axis=1).data
	dtas2000_L = dT.dtas2000.mean(axis=1).data
	dxax_L = dT.dxaxglob.mean(axis=1).data
	dxax1861_L = dT.dxax1861.mean(axis=1).data
	dxax2000_L = dT.dxax2000.mean(axis=1).data
	latwts = dT.Awts.sum(axis=1)
	
	# prepare figure
	fig = plt.figure()
	
	# axis 1: plot zonal average
	ax1 = fig.add_subplot(1,2,1,xlabel='Latitude',title='Latitude average',
						  xlim=[-90,90],ylabel='$\Delta$ T ($^\circ$C)')
	
	ax1.plot(lats,dtas_L,label='global coverage')
	ax1.plot(lats,dtas1861_L,label='1861-1880 coverage')
	ax1.plot(lats,dtas2000_L,label='2000-2009 coverage')
	
	leg1 = ax1.legend(loc='upper left',frameon=False)
	
	# axis 2: plot area-weighted zonal average
	ax2 = fig.add_subplot(1,2,2,xlabel='Latitude',xlim=[-90,90],
						  ylabel='$\Delta$ T ($^\circ$C)/5$^\circ$',
						  title='Area-weighted latitude average')
	ax2.yaxis.tick_right()
	ax2.yaxis.set_label_position("right")
	
	ax2.plot(lats,dtas_L*latwts,label='global')
	ax2.plot(lats,dtas1861_L*latwts,label='1861-1880 coverage')
	ax2.plot(lats,dtas2000_L*latwts,label='2000-2009 coverage')
	
	fig.set_size_inches(10,5)
	fig.subplots_adjust(0.08,0.1,0.9,0.95,wspace=0.03)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_zonebias_contribution(zonebias_dataloc,saveloc):
	""" Supplementary Figure 8.
	
	Cumulative zonal contribution to reported temperature change bias from 
	1861--1880 to 2000--2009 using masks from both periods, defined relative
	to unmasked data.
	"""
	
	# load data and generate handles, d2000 is difference of 2000 mask from
	# unmasked
	zonaldata = np.genfromtxt(zonebias_dataloc,skip_header=1)
	lats = zonaldata[:,0]
	d2000 = zonaldata[:,2] - zonaldata[:,1]
	d1861 = zonaldata[:,3] - zonaldata[:,1]
	
	# prepare and plot figure
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1,xlim=[-90,90],xlabel='Latitude',
						 ylabel='Cumulative $\Delta$T bias ($^\circ$C)')
	
	ax.plot(lats,d1861,label='1861 mask difference')
	ax.plot(lats,d2000,label='2000 mask difference')
	
	leg = plt.legend(loc='lower left',frameon=False)
	
	fig.set_size_inches(7,5)
	fig.subplots_adjust(0.13,0.1,0.95,0.95)
	
	if saveloc != None:
		fig.savefig(saveloc)


def plot_TCR_PDFs(dat,TCRfile,saveloc,sampleN=1e6):
	""" Supplementary Figure 10.
	"""
	
	# Get dT distributions
	T0 = delta61_09( dat.Tobs['HadCRUT4']['t'],dat.Tobs['HadCRUT4']['T'])
	dT = np.random.randn(sampleN) * 0.20/1.645 + T0
	# Get inputs for forcing from LC15 code output
	F0 = delta61_09(dat.RF['otto']['t'],dat.RF['otto']['F'])
	dFin = np.genfromtxt('../data/forcing/dF.dat')[:sampleN]
	dF2xin = np.genfromtxt('../data/forcing/dF2xCO2.dat')[:sampleN]
	
	# generate alpha correction factor using modelled TCR values
	modTCRs = np.genfromtxt(TCRfile,skip_header=4)[:,1:]
	fit = curve_fit(linfunc,modTCRs[:,2],modTCRs[:,0])
	alpha = np.random.choice(modTCRs[:,0] / modTCRs[:,2],sampleN)
	
	# open output file to write to
	dF,tcrpdf = internal_dF_TCR_calc(dT,dFin,dF2xin,F0,F2x0=3.44,DF2x=0.344,
										  DFscalefactor=1.)
	atcrpdf = alpha * tcrpdf
	
	# now produce figure
	fig = plt.figure()
	ax = fig.add_subplot(111,xlim=[0,4])
	
	ax.hist(tcrpdf,bins=np.linspace(0,4,500),histtype='step',normed=True,
			label='blended-masked')
	ax.hist(atcrpdf,bins=np.linspace(0,4,500),histtype='step',normed=True,
			label='tas-only')
	
	leg = ax.legend(frameon=False,loc='upper right')
	
	for ltext in leg.get_texts():
		ltext.set_fontsize(15)
	
	for xtl in ax.get_xticklabels():
		xtl.set_fontsize(15)
	for ytl in ax.get_yticklabels():
		ytl.set_fontsize(15)
	ax.set_xlabel('TCR ($^\circ$C)',fontsize=20)
	ax.set_ylabel('P density',fontsize=20)
	
	fig.subplots_adjust(0.1,0.1,0.95,0.95)
	fig.savefig(saveloc)
