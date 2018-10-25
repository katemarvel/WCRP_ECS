#!/usr/bin/env python
# coding: utf-8

# (c) British Crown Copyright, the Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following 
# conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the Met Office nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
# THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MPL Plot

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy import stats

# normal distribution confidence scaling factors
# confidence Range  [1-p,p](%)  Z-score = +/- n sd range 
#                                                           
# 95%        [  2.5  , 97.5  ]  1.959
# 90%        [  5.   , 95.   ]  1.64        
# 80%        [ 10    , 90    ]  1.28
# 60%        [ 20    , 80    ]  0.84

def update_ecs(chi,epsilon,prior,bin_centres):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()
  # update ECS pdf using SSBW chi and epsilon values 
  shape = prior.shape
  likelihood=np.zeros(shape[0])
  erf_width=2.0 # parameter controlling rate of transition between likelihood min and max values
  #erf_width=1000.0 # convert to step function
  for x in range(0, shape[0]):
      # SSBW likelihood
      likelihood[x] = (( 1 - 2 * epsilon ) * math.erf(erf_width*bin_centres[x]-erf_width*chi)+1)/2

      # alternative likelihood which ramps up to 1 rather than (1-epsilon)
      # if chi < 3:   # LS condition
      #     likelihood[x] =  1 - ( (1-epsilon)*(1-(math.erf(erf_width*bin_centres[x]-erf_width*chi)+1)/2) )
      # if chi > 3:   # HS condition
      #     likelihood[x] =  (1-epsilon) + epsilon*(1-(math.erf(erf_width*bin_centres[x]-erf_width*chi)+1)/2)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  plt.plot(bin_centres,likelihood,linewidth=2,color=color)
  plt.text(xpos,0.75,' SSBW condition',color=color)
  
  return (likelihood,posterior)

def update_ecs_palaeo1(prior,bin_centres):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  from io import StringIO   # StringIO behaves like a file object
  palaeo= np.loadtxt('palaeo_likelihood.txt',skiprows=1)
  print palaeo.shape

  # update ECS pdf using palaeo group evidence

  shape = prior.shape
  #likelihood=np.zeros(shape[0])
  likelihood=np.interp(bin_centres,palaeo[:,0],palaeo[:,1])

  likelihood[likelihood < 0.02] = 0.02

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def update_ecs_palaeo2(prior,bin_centres):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  from io import StringIO   # StringIO behaves like a file object
  palaeo= np.loadtxt('palaeo_likelihood.txt',skiprows=1)
  print palaeo.shape

  # update ECS pdf using palaeo group evidence

  shape = prior.shape
  #likelihood=np.zeros(shape[0])
  likelihood=np.interp(bin_centres,palaeo[:,0],palaeo[:,1])

  likelihood[likelihood < 0.02] = 0.02

  likelihood=np.sqrt(likelihood)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def update_ecs_piers_textfile(prior,bin_centres):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  from io import StringIO   # StringIO behaves like a file object
  piers= np.loadtxt('ECS_pdf_corrected.txt')
  print piers.shape

  # update ECS pdf using piers text file 

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape
  #likelihood=np.zeros(shape[0])
  likelihood=np.interp(bin_centres,piers[:,0],piers[:,2])*100.0

  #likelihood[likelihood < 0.02] = 0.02

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  plt.plot(bin_centres,likelihood,linewidth=2,color=color)
  #plt.plot(piers[:,0],piers[:,1]*150.,color=color)
  #plt.plot(piers[:,0],piers[:,2]*150.,color=color)
  #plt.text(xpos,0.55,'Historical #3 piers col 3 interpolation',color=color)
  #plt.text(xpos,0.55,'Historical #3 before interpolation',color=color)
  plt.text(xpos,0.6,'Historical #3',color=color)
  
  return (likelihood,posterior)

def update_ecs_kasia(prior,bin_centres,n_samples,bin_boundaries,transfer,likelihood_shape):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  shape = prior.shape
#  kasia_samples_s=np.random.gamma(30.5,0.086,n_samples) # Johansson et al 2015
#  kasia_samples_s=np.random.gamma(4.415,0.444,n_samples) # Skeie et al 2014
#  ecs_hist=np.random.gamma(9.8,0.226,n_samples) # synthetic ECS

# 95%        [  2.5  , 97.5  ]  1.959
# 90%        [  5.   , 95.   ]  1.64        
# 80%        [ 10    , 90    ]  1.28
# 60%        [ 20    , 80    ]  0.84

  # Choose Gaussian distribution with [10-90] (80%) confidence interval of 1.4-3.2K  
  #kasia_shist_mean=(3.2+1.4)/2
  #kasia_shist_50_to_90_range=(3.2-1.4)/2
  #kasia_shist_sd=kasia_shist_50_to_90_range/1.28   #  1.28 is z-score for 80% (10-90) confidence
  #ecs_hist=np.random.normal(kasia_shist_mean, kasia_shist_sd,n_samples)

  match=0

  if likelihood_shape == 'gaussian':
      # Choose Gaussian distribution with [20-80] (60%) confidence interval of 1.4-3.2K  
      # This is the same as the [10-90] confidence interval of 1.4-3.2K but with the 5% tail probabilities doubled
      kasia_shist_mean=(3.2+1.4)/2
      kasia_shist_50_to_90_range=(3.2-1.4)/2
      kasia_shist_sd=kasia_shist_50_to_90_range/0.84   #  0.84 is z-score for 60% (20-80) confidence
      ecs_hist=np.random.normal(kasia_shist_mean, kasia_shist_sd,n_samples)
      match=1

  if likelihood_shape == 'wald':
      # wald is inverse gamma distribution
      # Kasia's parameters
      # ecs_hist=np.random.wald(2.23,  20.6 ,n_samples) # synthetic ECS
      # Mark's fit
      ecs_hist=np.random.wald(2.2278,20.5896,n_samples) # synthetic ECS
  match=1

  if match == 0:
    print "likelihood_shape=",likelihood_shape
    print "transfer=",transfer
    sys.exit()

  ecs_hist=historical_transfer_function(ecs_hist,transfer)

#  kasia_samples_s= transfer_armour2017_mc(kasia_samples_s,'armour2017_table_s1.txt',(3,2))
#  kasia_samples_s = transfer_armour2017_mc(kasia_samples_s,'armour2017_table_s1_and_ar5.txt',(3,4))
#  kasia_samples_s = transfer_armour2017_mc(kasia_samples_s,'armour2017_table_s1_avg_similar.txt',(3,2))

#  kasia_samples_s = transfer_alpha_historical(kasia_samples_s,hist_f2co2,'andrews_amip_piforcing_inprep.txt')
  
  likelihood, bin_boundaries = np.histogram(ecs_hist, bins=bin_boundaries,density=True)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)
  
def transfer_alpha_historical(s,fos,file):

  alpha=fos/s

  scaling_alphas= np.loadtxt(file,skiprows=1,usecols=(1,2))

  alpha_historical=scaling_alphas[:,0]
  alpha_4xco2=scaling_alphas[:,1]

  scaling=alpha_4xco2/alpha_historical

  scalings=np.random.choice(scaling,size=np.shape(s))

  alpha2=alpha*scalings

  s2=fos/alpha2

  #plt.scatter(s[1:1000],s2[1:1000])
  #plt.show()

  return (s2)

def transfer_scale_s_by_normal(s,mu_s_scaling,sigma_s_scaling):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  scaling_on_s=mu_s_scaling+sigma_s_scaling*np.random.normal(0,1,n_samples)

  s2 = s * scaling_on_s

  return (s2)

def transfer_scale_l_by_normal(s,mu_l_scaling,sigma_l_scaling):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  scaling_on_l=mu_l_scaling+sigma_l_scaling*np.random.normal(0,1,n_samples)

  s2 = s / scaling_on_l

  return (s2)

def transfer_armour2017_mc_old(s,file,cols):

  # old version scales with randomly selected models read from a table. Superseded
  # be my approach which is to sample from a PDF fitted to model values mean and variance

  #armour_ecs= np.loadtxt('armour2017_table_s1.txt',skiprows=1,usecols=(3,2))
  armour_ecs= np.loadtxt(file,skiprows=1,usecols=cols)
  print armour_ecs.shape

  ecs_infer=armour_ecs[:,0]
  ecs21_120=armour_ecs[:,1]

  scaling=ecs21_120/ecs_infer

  scalings=np.random.choice(scaling,size=np.shape(s))

  s2=s*scalings

  #plt.scatter(s[1:1000],s2[1:1000])
  #plt.show()

  return (s2)

def update_ecs_piers_document(prior,bin_centres,n_samples,bin_boundaries,transfer,normal01_2co2):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # attempt to code up Piers' calculation based on his document

  f2co2_hist=f2co2_historical(normal01_2co2)

  # assume erf and f2co2_hist are correlated
  # median quoted is 2.33 but we want to get tails right so use 2.285=(3.10+1.47)/2
  erf=2.33+normal01_2co2*(3.10-1.47)/2/1.64    # updated 28/3/18
  # subtract 0.05 to get better fit to Piers's erf percentiles 5/4/18
  # Piers's IDL total ERF      1.46898      2.33295      3.10208
  # We Assume erf and f2co2_hist are perfectly correlated by basing both on normal01_2co2
  # Not exactly what Piers did
  erf=2.33+normal01_2co2*(3.10-1.47)/2/1.64-.05   
  # gives erf [5,50,95] percentiles [1.46, 2.28, 3.10] which is a good match with Piers's [1.47, 2.33 ,3.10]

  # n is TOA radiation 
  n=np.random.normal(0.61,(0.14)/1.64,n_samples) # confirmed 28/3/18

  T=np.random.normal(0.91,(0.11)/1.64,n_samples) # confirmed 28/3/18

  # apply fudge factor to T to get similar range to Piers's numbers for shist.
  # Need to do this because I don't currently have access to the correlation structure
  # between erf_hist and f2co2 in Piers's idl save file

  n=n-0.61
  n=n*1.2
  n=n*1.5
  n=n*2
  n=n+0.61

  # Piers's IDL code
  #Tobs = 0.91
  #Q=0.61+p1 * 0.14/1.64 ; 90% range 0.14 assume normal distributed
  #e=  p2 * 0.11/1.64 ; 90% range 0.11 assume normal distributed

  ecs_hist=f2co2_hist*T/(erf-n)

  # apply fudge factor to improve agreement with Piers's calculations

  ecs_hist=ecs_hist*1.1
  ecs_hist=(ecs_hist-2.3)*1.2+2.3

  # shist [ 1.60 ,       2.08 ,       3.36 ] Otto-like calc

  # Piers  5%, 50% and 95% Shist      1.36952      1.99892      3.84180 (old method no scaling)
  #  This program:
  #  1.54 ,       2.08 ,       3.42
  #  1.60 ,       2.10 ,       3.40   
  #  1.48 ,       2.08 ,       4.08  (this comparison not so important as we're trying to approximate new method)

  # Piers [20-50-80] range for Shist from document = [1.75,2.25,3.15] (new method no scaling)
  # This program gives [20-50-80] range  = [ 1.8,  2.30 ,   3.14 ] (new method no scaling)

  # Alternative approach - use my code which reproduces Otto and rescale to Piers's numbers 

  # Attempt to reproduce otto et al calculation

  otto=0
  if otto:
  
      norm1=np.random.normal(0,1,n_samples)
      norm2=np.random.normal(0,1,n_samples)
      norm3=np.random.normal(0,1,n_samples)
      norm4=np.random.normal(0,1,n_samples)
      norm5=np.random.normal(0,1,n_samples)
      norm6=np.random.normal(0,1,n_samples)
    
      fco2_historical_forcing_mean=2.83            # Values for 2010 from page 3 para 4 Otto et al
      fco2_historical_forcing_50_to_95_range=0.28
      # Assume fco2_historical and f2co2 correlated by using norm1 for both
      fco2_historical_forcing=fco2_historical_forcing_mean+norm1*fco2_historical_forcing_50_to_95_range/1.64 
    
      natural_historical_forcing_mean=0.17         # Values for 2003 from page 3 para 4 - may need adjustment Otto et al
      natural_historical_forcing_50_to_95_range=0.2
      natural_historical_forcing=natural_historical_forcing_mean+norm2*natural_historical_forcing_50_to_95_range/1.64 
    
      residual_historical_forcing_mean=1.05         # Values for 2010 from page 3 para 4 Otto et al
      residual_historical_forcing_50_to_95_range=0.72
      residual_historical_forcing=residual_historical_forcing_mean+norm3*residual_historical_forcing_50_to_95_range/1.64 
    
      total_historical_forcing=fco2_historical_forcing+natural_historical_forcing+residual_historical_forcing
    
      # scale to match desired mean and stdev from Piers's calculations
    
      f2co2_mean=3.8
      f2co2_50_to_95_range=0.74
      f2co2=f2co2_mean+norm1*f2co2_50_to_95_range/1.64 
    
      #erf=2.33+normal01_2co2*(3.10-1.47)/2/1.64-.05 # Numbers from Piers but with adjustment for non-normality to improve ranges
      total_historical_forcing_mean=2.33-0.05 
      total_historical_forcing_50_to_95_range=(3.10-1.47)/2
    
      normalised_forcing=(total_historical_forcing-np.mean(total_historical_forcing))/np.std(total_historical_forcing)
      total_historical_forcing=normalised_forcing*total_historical_forcing_50_to_95_range/1.64+total_historical_forcing_mean
    
      # Piers's IDL code
      #Tobs = 0.91
      #Q=0.61+p1 * 0.14/1.64 ; 90% range 0.14 assume normal distributed
      #e=  p2 * 0.11/1.64 ; 90% range 0.11 assume normal distributed
    
      t_mean = 0.91
      t_50_to_95_range = 0.11
      T=t_mean+norm4*t_50_to_95_range/1.64 
    
      n_mean = 0.61
      n_50_to_95_range = 0.14
      n=n_mean+norm5*n_50_to_95_range/1.64 
    
      ecs_hist=f2co2_hist*T/(erf-n)
    
      # ERF_2xCO2      3.02272      3.80017      4.53098
      # total ERF      1.46898      2.33295      3.10208
      # Tobs+e     0.799676     0.910284      1.02006
      # Q     0.469839     0.610587     0.751300
      # Shist  5%, 50% and 95% Shist      1.36952      1.99892      3.84180
    
      # This program
    
      # Shist  5%, 50% and 95% Shist    1.48 ,       2.08 ,       4.10 
      # f2co2_hist [3.06, 3.80, 4.54]  GOOD
      # T percentiles [0.79, 0.91, 1.02] GOOD
      # n percentiles [0.47, 0.61, 0.75] GOOD
      # erf percentiles [1.46, 2.28, 3.09] GOOD
      # shist [ 1.60 ,       2.08 ,       3.36 ] Otto-like calc
    
  np.set_printoptions(precision=2)

  print 'f2co2_hist', np.percentile(f2co2_hist,[5,50,95])

  print 'T percentiles', np.percentile(T,[5,50,95])
  print 'n percentiles', np.percentile(n,[5,50,95])
  print 'erf percentiles', np.percentile(erf,[5,50,95])
  print 'ecs_hist percentiles', np.percentile(ecs_hist,[5,50,95])
  #piers_samples_gt1=piers_samples[piers_samples > 1]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape

  ecs_hist=historical_transfer_function(ecs_hist,transfer)

  likelihood, bin_boundaries = np.histogram(ecs_hist, bins=bin_boundaries,density=True)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def update_ecs_piers_document_old(prior,bin_centres,n_samples,bin_boundaries,transfer,normal01_2co2):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # attempt to code up Piers' calculation based on his document

  f2co2_hist=f2co2_historical(normal01_2co2)

  # assume erf and f2co2_hist are correlated
  # median quoted is 2.33 but we want to get tails right so use 2.285=(3.10+1.47)/2
  erf=2.285+normal01_2co2*(3.10-1.47)/2/1.64    # updated 28/3/18

  # assume erf and f2co2_hist are uncorrelated
  #erf=np.random.normal(2.29,(3.06-1.43)/2/1.64,n_samples)

  # n is TOA radiation 
  n=np.random.normal(0.61,(0.14)/1.64,n_samples) # confirmed 28/3/18
  #n=np.random.normal(0.81,(0.14)/1.64,n_samples) # incorrect version to reproduce values in document

  T=np.random.normal(0.91,(0.11)/1.64,n_samples) # confirmed 28/3/18
  #f=np.random.normal(3.7,(0.74)/1.64,n_samples)

  ecs_hist=f2co2_hist*T/(erf-n)

  #piers_samples_gt1=piers_samples[piers_samples > 1]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape

  ecs_hist=historical_transfer_function(ecs_hist,transfer)

  likelihood, bin_boundaries = np.histogram(ecs_hist, bins=bin_boundaries,density=True)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def historical_transfer_function(ecs_hist,transfer):

  not_applied=1

  if transfer == 'none':
    ecs=ecs_hist
    not_applied=0

  if transfer == 'armour':
#   a= 1.26+p3*0.2 ; Armour scaling for S/Shist 1.26 with 0.2 standard deviation
    ecs= transfer_scale_s_by_normal(ecs_hist,1.26,0.2)
    not_applied=0

  if transfer == 'andrews':
#   tim= 0.62+p3*0.13 ; Andrews scaling for lambda of 0.62 with 0.13 standard deviation
    ecs= transfer_scale_l_by_normal(ecs_hist,0.62,0.13)
    not_applied=0

#  ecs_hist= transfer_armour2017_mc(ecs_hist,'armour2017_table_s1.txt',(3,2))
#  ecs_hist = transfer_armour2017_mc(ecs_hist,'armour2017_table_s1_and_ar5.txt',(3,4))
#  ecs_hist = transfer_armour2017_mc(ecs_hist,'armour2017_table_s1_avg_similar.txt',(3,2))

  if not_applied ==1:
    print "unknown transfer function"
    sys.exit()

  return(ecs)

def update_ecs_piers_eyeball(prior,bin_centres,n_samples,bin_boundaries,transfer,normal01_2co2):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # eyeball fit to Piers pdf using ratio of two gaussians - happens to match process group likelihood!
  #fomu,fosigma=3.7,0.5   # use same F as process group
  #fos=np.random.normal(fomu,fosigma,n_samples)
  femu,fesigma = -1.35, 0.4 # mean and sigma

  fes=np.random.normal(femu,fesigma,n_samples)

  print 'mean forcing = ',np.mean(fos, dtype=np.float64)

  piers_samples=-1*fos/fes

  # remove any ecs < 1
  piers_samples_gt1=piers_samples[piers_samples > 1]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape
  likelihood, bin_boundaries = np.histogram(piers_samples_gt1, bins=bin_boundaries,density=True)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  plt.plot(bin_centres,likelihood,linewidth=2,color=color)
  plt.text(xpos,0.6,'Historical #2',color=color)
  
  return (likelihood,posterior)

def f2co2_process(normal01_2co2):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # values from Steve Klein:
  fomu,fosigma=3.7,0.3
  f2co2_proc=fomu+normal01_2co2*fosigma
  #f2co2_proc=np.random.normal(fomu,fosigma,n_samples)

  return f2co2_proc

def f2co2_historical(normal01_2co2):

  # based on numbers from piers' document
  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()
  #normal01_2co2=np.random.normal(0,1,n_samples)
  f2co2_hist=3.8+normal01_2co2*0.74/1.64  # updated 28/3/18
  # fudged numbers to get better agreemennt with piers's calcs, which use values read in from idl save file 
  # piers percentiles 5%, 50% and 95% Shist                1.36952      1.99892      3.84180
  # numbers I get using fitting gaussians to Piers's ranges: 1.28 ,       2.08 ,       4.20
  #f2co2_hist=3.8+normal01_2co2*0.74/1.64*0
  #ecs_hist percentiles [1.3239847438079446, 2.0101364905651788, 3.9124436590599641]

  return f2co2_hist

def lambda_process():
  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  femu,fesigma = -1.3, 0.42 # mean and sigma
  fes=np.random.normal(femu,fesigma,n_samples)

  return fes

def update_ecs_process1(prior,bin_centres,n_samples,bin_boundaries,normal01_2co2):

  fes=lambda_process()

  f2co2_proc=f2co2_process(normal01_2co2)

  ecss_all=-1*f2co2_proc/fes

  # remove any negative ecs
  ecss=ecss_all[ecss_all > 0]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape
  likelihood, bin_boundaries = np.histogram(ecss, bins=bin_boundaries,density=True)

  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def update_ecs_process_s_minus_2(prior,bin_centres,n_samples,bin_boundaries,normal01_2co2):

  # same as process1 but with s^-2 prior backed out.

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  fes=lambda_process()

  f2co2_proc=f2co2_process(normal01_2co2)

  ecss_all=-1*f2co2_proc/fes

  # remove any negative ecs
  ecss=ecss_all[ecss_all > 0]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape
  process_likelihood, bin_boundaries = np.histogram(ecss, bins=bin_boundaries,density=True)

  prior=calc_s_semi_inf_lambda_prior(n_samples)

  likelihood = process_likelihood
  posterior = likelihood

  likelihood=np.copy(process_likelihood)
  likelihood[prior > 0] = process_likelihood[prior > 0]/(prior[prior > 0])
  likelihood[prior <= 0] = 0

  posterior_numerator = likelihood 

  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def update_ecs_process2(prior,bin_centres,n_samples,bin_boundaries,normal01_2co2):

  # same as process1 but with ssbw-like prior backed out.

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  fes=lambda_process()

  f2co2_proc=f2co2_process(normal01_2co2)

  ecss_all=-1*f2co2_proc/fes

  # remove any negative ecs
  ecss=ecss_all[ecss_all > 0]

  bin_width=bin_centres[1]-bin_centres[0]
  shape = prior.shape
  process_likelihood, bin_boundaries = np.histogram(ecss, bins=bin_boundaries,density=True)

  #ssbw-like prior as defined in plot_ecs_pdf
  #prior_label='SSBW-like Prior (-1.8,0.9,3.7,0.74)'

  prior_label='SSBW-like Prior F~N(3.7,0.74) L~N(-1.8,0.9))'
  ssbw_prior=calc_ssbw_prior(-1.8,0.9,3.7,0.74,prior_label,n_samples)

  likelihood = process_likelihood
  posterior = likelihood

  print (ssbw_prior.shape)
  print (process_likelihood.shape)
  likelihood=np.copy(process_likelihood)
  likelihood[ssbw_prior > 0] = process_likelihood[ssbw_prior > 0]/(ssbw_prior[ssbw_prior > 0])
  likelihood[ssbw_prior <= 0] = 0

  posterior_numerator = likelihood 

  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def normalise_pdf(pdf,bin_width):
  return(pdf/np.sum(pdf, dtype=np.float64)/bin_width)

  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

def ecs_cdf(count,bin_boundaries,bin_width,label,slabel):
  cdf=np.float64(0.0)
  ecs_mean=np.float64(0.0)
  ecs_mode=np.float64(0.0)
  ecs_weight=np.float64(0.0)
  # ubin_boundaries contains upper limits of bin_boundaries 
  ubin_boundaries=bin_boundaries+bin_width
  shape = count.shape
  ecs5=0.0
  ecs10=0.0
  ecs17=0.0
  ecs20=0.0
  ecs50=0.0
  ecs80=0.0
  ecs83=0.0
  ecs90=0.0
  ecs95=0.0
  p_ecs_gt_3=0.0
  
  for x in range(0, shape[0]):
           bin_centre=ubin_boundaries[x]+bin_width/2
	   cdf=cdf+count[x]*bin_width*0.5
           ecs_mean=ecs_mean+bin_centre*count[x]
           ecs_weight=ecs_weight+count[x]
	   if bin_centre < 2:
	   	 p_ecs_lt_2=cdf*100
	   if bin_centre < 1:
	   	 p_ecs_lt_1=cdf*100
	   if bin_centre < 1.5:
	   	 p_ecs_lt_1p5=cdf*100
	   if bin_centre < 0:
	     p_ecs_lt_0=cdf*100
	   if bin_centre > 3 and p_ecs_gt_3 == 0:
             p_ecs_gt_3=cdf*100
	   if bin_centre < 3:
             p_ecs_lt_3=cdf*100
	   if bin_centre < 4:
             p_ecs_lt_4=cdf*100
	   if bin_centre < 4.5:
             p_ecs_lt_4p5=cdf*100
	   if bin_centre < 6.0:
             p_ecs_lt_6=cdf*100
           if cdf > 0.05 and ecs5 == 0:
             ecs5=bin_boundaries[x]+bin_width/2
           if cdf > 0.17 and ecs17 == 0:
             ecs17=bin_boundaries[x]+bin_width/2
           if cdf > 0.1 and ecs10 == 0:
             ecs10=bin_boundaries[x]+bin_width/2
           if cdf > 0.2 and ecs20 == 0:
             ecs20=bin_boundaries[x]+bin_width/2
           if cdf > 0.5 and ecs50 == 0:
             ecs50=bin_boundaries[x]+bin_width/2
           if cdf > 0.83 and ecs83 == 0:
             ecs83=bin_boundaries[x]+bin_width/2
           if cdf > 0.8 and ecs80 == 0:
             ecs80=bin_boundaries[x]+bin_width/2
           if cdf > 0.90 and ecs90 == 0:
             ecs90=bin_boundaries[x]+bin_width/2
           if cdf > 0.95 and ecs95 == 0:
             ecs95=bin_boundaries[x]+bin_width/2
	   cdf=cdf+count[x]*bin_width*0.5
  p_ecs_gt_4p5=1-p_ecs_lt_4p5

  n=np.shape(bin_boundaries)
  n=n[0]-1
  print "n=",n
  ecs_mean=ecs_mean/ecs_weight

  ecs_mode_index=np.argmax(count)
  ecs_mode=bin_boundaries[ecs_mode_index]+bin_width/2

  print 'bin_width = ',bin_width

  print "Prior;P(ECS <1);P(ECS <1.5);P(ECS <2);P(ECS >4);P(ECS >4.5);P(ECS >6);P(1.5 -4.5);5th %ile;10th %ile;17th %ile;20th %ile;50th %ile;80th %ile;83rd %ile;90th %ile;95th %ile;Mode;Mean"
  
  qlabel='"'+label+'"'
  f='10.2f'
  print format(qlabel,'10s'),';',format( p_ecs_lt_1,f),';',format( p_ecs_lt_1p5,f),';',format( p_ecs_lt_2,f),';',format( 100-p_ecs_lt_4,f),';',format( 100-p_ecs_lt_4p5,f),';',format( 100-p_ecs_lt_6,f),';',format( p_ecs_lt_4p5-p_ecs_lt_1p5,f),';',format( ecs5,f),';',format( ecs10,f),';',format( ecs17,f),';',format( ecs20,f),';',format( ecs50,f),';',format( ecs80,f),';',format( ecs83,f),';',format( ecs90,f),';',format( ecs95,f),';',format( ecs_mode,f),';',format( ecs_mean,f)

  print
#  print 'P(ECS < 0) = ' , p_ecs_lt_0
#  print 'P(ECS < 1.5) = ' , p_ecs_lt_1p5
#  print 'P(ECS < 2.0) = ' , p_ecs_lt_2
#  print 'P(ECS < 3.0) = ' , p_ecs_lt_3
#  print 'P(ECS > 3.0) = ' , p_ecs_gt_3
#  print 'P(ECS > 4.0) = ' , 1-p_ecs_lt_4
#  print 'P(ECS > 4.5) = ' , 1-p_ecs_lt_4p5
#  print 'P(ECS > 6.0) = ' , 1-p_ecs_lt_6
#  print 'P(1.5 < ECS < 4.5) = ' , p_ecs_lt_4p5 - p_ecs_lt_1p5
#  print '5th percentile = ' , ecs5
#  print '50th percentile = ' , ecs50
#  print '95th percentile = ' , ecs95
#  print '5th minus 50th = ' , ecs5-ecs50
#  print '95th minus 50th = ' , ecs95-ecs50
  #print 'ptest' , ptest
  print 'histogram total' , cdf
  print 'cdf=', cdf
  assert (cdf > 0.99)
  assert (cdf <= 1.01)

  slabel2=slabel+'.txt'
  slabel2=slabel2.replace(' ','_')
  slabel2=slabel2.replace('-','_')
  slabel2=slabel2.replace('/','_')

  #file = open(slabel2,"w") 
  #file.write(bin_boundaries)
  #file.write(count)
  #file.close() 

  return (ecs5,ecs50,ecs95)

def calc_ssbw_prior(femu,fesigma,fomu,fosigma,label,n_samples):

#  label='Stevens et al 2016'
#  fomu,fosigma=3.7,3.7*0.2
#  femu,fesigma = -1.6, 1.6*0.5 # mean and sigma

  import os.path

  f='ssbw-prior.txt'
  if os.path.isfile(f):
      prior=np.loadtxt(f)
  else:
      (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()
    
      fes=np.random.normal(femu,fesigma,n_samples)
      f2co2_ssbw=np.random.normal(fomu,fosigma,n_samples)
    
      print 'mean forcing = ',np.mean(f2co2_ssbw, dtype=np.float64)
      
      ecss_all=-1*f2co2_ssbw/fes
    
      # remove any negative ecs
      ecss=ecss_all[ecss_all > 0]
    
      # Exclude ECS > 10 to reproduce Bjorn's calculations in SSBW
      #ecss2=ecss[ecss < 10]
      #ecss=ecss2
    
      prior_label=label
    
      prior, bin_boundaries = np.histogram(ecss, bins=bin_boundaries,density=True)
      prior=normalise_pdf(prior,bin_width)
      # put small non-zero value in to help to back put prior in a reversible way
      prior[prior <= 0] = 1e-12
      shape = prior.shape
    
      np.savetxt(f,prior)

  return(prior)

def calc_uniform_s_prior(lower,upper,n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  prior_samples=np.random.uniform(lower,upper,n_samples)

  prior, bin_boundaries = np.histogram(prior_samples, bins=bin_boundaries,density=True)

  return(prior)

def calc_s_semi_inf_lambda_prior(n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  prior = bin_centres*0

  prior[bin_centres > 0.0] = 10*bin_centres[bin_centres > 0.0]**-2

  prior=normalise_pdf(prior,bin_width)

  return(prior)

def calc_cauchy_prior(loc,scale,n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  prior_samples = stats.cauchy.rvs(loc=loc, scale=scale, size=n_samples)

  prior_samples_positive=prior_samples[prior_samples > 0]
  prior_samples_0_100=prior_samples_positive[prior_samples_positive < 100]

  prior, bin_boundaries = np.histogram(prior_samples_0_100, bins=bin_boundaries,density=True)

  return(prior)

import numpy as np
def rejection_sampler(p,xbounds,pmax):
    while True:
        x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
        y = np.random.rand(1)*pmax
        if y<=p(x):
            return x

def calc_cauchy_like_prior_ah09(cauchy_location,cauchy_scale,n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  #uniform_sample=np.random.uniform(0,100,n_samples)
  #plot cauchy prior density using formula from annan and hargreaves 2009
  #prior_samples=1.0/((uniform_sample-cauchy_location)**2+cauchy_scale)
  
  prior=1.0/((bin_centres-cauchy_location)**2+cauchy_scale)
  prior[bin_centres < 0]=0
  prior[bin_centres > 100]=0

  normalised_prior=normalise_pdf(prior,bin_width)

  return(normalised_prior)

def calc_uniform_f_lambda_prior(folower,foupper,felower,feupper,n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  fosamples=np.random.uniform(folower,foupper,n_samples)
  fesamples=np.random.uniform(felower,feupper,n_samples)
  prior_samples=-1*fosamples/fesamples

  prior, bin_boundaries = np.histogram(prior_samples, bins=bin_boundaries,density=True)

  return(prior)

def calc_normal_f_uniform_lambda_prior(fomu,fosigma,felower,feupper,n_samples):

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  fosamples=np.random.normal(fomu,fosigma,n_samples)
  fesamples=np.random.uniform(felower,feupper,n_samples)
  prior_samples=-1*fosamples/fesamples

  prior, bin_boundaries = np.histogram(prior_samples, bins=bin_boundaries,density=True)

  return(prior)

def plot_likelihood(label,color,pos,tickscale,normal01_2co2,uprior,exponent,s_exponent):

  xpos=8
  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()
  ticklen=0.05

  (likelihood,normalised_likelihood,posterior)=likelihood_update(label,normal01_2co2,uprior,uprior,exponent)

#  plt.plot(bin_centres,normalised_likelihood,color,linewidth=2)
  # plot is normalised to max likelihood in range [0,20] because poor sampling in the tails can give larger values in the case of process L2
  likelihood_plotted=likelihood[bin_centres < 20]
  plt.plot(bin_centres,likelihood/likelihood_plotted.max(),color,linewidth=2)
  plt.text(xpos,0.5-0.05*(pos+1),label+s_exponent,color=color)

  (ecs5,ecs50,ecs95)=ecs_cdf(normalised_likelihood,bin_boundaries,bin_width,label,label)

  plt.plot([ecs5,ecs5],[0,ticklen*tickscale],linewidth=2,color=color)
  plt.plot([ecs95,ecs95],[0,ticklen*tickscale],linewidth=2,color=color)
  plt.plot([ecs50,ecs50],[0,ticklen*tickscale*2],linewidth=2,color=color)

  print ("pos=",pos)
  pos=pos-0.05
  print ("pos=",pos)

def plot_likelihoods():

  import os.path

  ticklen=0.05
  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # normalise likelihoods by combining them with a uniform prior over -100,100
  f='u100.txt'
  if os.path.isfile(f):
      uprior=np.loadtxt(f)
  else:
      uprior=calc_uniform_s_prior(0,100,n_samples)
      np.savetxt(f,uprior)

  normal01_2co2='now read in at lower level' 

  xrange=[-0.5,14]
  yrange=[0,1.1]
  plt.xlim(xrange)
  plt.ylim(yrange)
  color='k'
  plt.plot([1.5,1.5],yrange,color=color)
  plt.plot([4.5,4.5],yrange,color=color)
  plt.plot([-100,100],[0.2,0.2],color='k')

  plot_palaeo_likelihoods=0
  plot_process_likelihoods=0
  plot_historical_likelihoods=0
  plot_main_likelihoods=1

  if plot_main_likelihoods:
#    plot_likelihood('Process L1','c',-6,1,normal01_2co2,uprior,1,'')
#    plot_likelihood('Process L2','m',-5,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Process Likelihood','r',-4,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical Likelihood','g',-3,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Palaeo Likelihood','b',-2,1,normal01_2co2,uprior,1,'')

  if plot_historical_likelihoods:

    plot_likelihood('Historical L1','r',-4,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L2','g',-3,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L3','b',-2,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L4','m',-1,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L5','c',0,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L6','y',1,1,normal01_2co2,uprior,1,'')
    plot_likelihood('Historical L3+L6','k',2,1,normal01_2co2,uprior,1,'')

  if plot_palaeo_likelihoods:

    plot_likelihood('Palaeo L1','r',-4,1,normal01_2co2,uprior,1,'')
#    plot_likelihood('Palaeo L2','g',2,1,normal01_2co2,uprior,1,'')
#    plot_likelihood('Palaeo L1-2','k',3,1,normal01_2co2,uprior,1,'')

  if plot_process_likelihoods:

#    prior_label='SSBW-like Prior F~N(3.7,0.74) L~N(-1.8,0.9))'
#    ssbw_prior=calc_ssbw_prior(-1.8,0.9,3.7,0.74,prior_label,n_samples)
#    plt.plot(bin_centres,ssbw_prior,linewidth=2,color='b')
#    plt.text(xpos,0.45,'SSBW-like Prior S=-F/L',color='b')

    plot_likelihood('Process L1','r',-4,1,normal01_2co2,uprior,1,'')

#    plot_likelihood('Process L2 (PDF^0.5)','g',2,1,normal01_2co2,uprior,1,'^0.5')

    plot_likelihood('Process L2','g',-3,1,normal01_2co2,uprior,1,'')

#    plt.text(xpos,0.42,'F~N(3.7,3.7*0.2) L~N(-1.8,1.8*0.5)',color='b')
  # plt.text(xpos,0.45,'SSBW Prior F~N(3.7,3.7*0.2) L~N(-1.6,1.6*0.5)',color=color)
  
#    plot_likelihood('Process L1-2','b',3,1,normal01_2co2,uprior,1,'')
  
  plt.show()

def likelihood_update(label,normal01_2co2,uprior,prior,exponent):

  import os.path

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  if os.path.isfile(label+'.normalised_likelihood.txt'):
      normalised_likelihood=np.loadtxt(label+'.normalised_likelihood.txt')
      likelihood=np.loadtxt(label+'.likelihood.txt')

  else:

      f='normal01_2co2.txt'
      if os.path.isfile(f):
          normal01_2co2=np.load(f)
      else:
          normal01_2co2=np.random.normal(0,1,n_samples) 
          np.save(f,normal01_2co2)

      if label == 'Historical Likelihood':
        (likelihood,normalised_likelihood)=update_average_likelihoods(uprior,['Historical L3','Historical L6'],normal01_2co2)
    
      if label == 'Historical L3+L6':
        (likelihood,normalised_likelihood)=update_average_likelihoods(uprior,['Historical L3','Historical L6'],normal01_2co2)
    
      if label == 'Historical L1':
        (likelihood,normalised_likelihood)=update_ecs_piers_document(uprior,bin_centres,n_samples,bin_boundaries,'none',normal01_2co2)
      
      if label == 'Historical L2':
        (likelihood,normalised_likelihood)=update_ecs_piers_document(uprior,bin_centres,n_samples,bin_boundaries,'armour',normal01_2co2)
    
      if label == 'Historical L3':
        (likelihood,normalised_likelihood)=update_ecs_piers_document(uprior,bin_centres,n_samples,bin_boundaries,'andrews',normal01_2co2)
    
      if label == 'Historical L4':
        (likelihood,normalised_likelihood)=update_ecs_kasia(uprior,bin_centres,n_samples,bin_boundaries,'none','wald')
    
      if label == 'Historical L5':
        (likelihood,normalised_likelihood)=update_ecs_kasia(uprior,bin_centres,n_samples,bin_boundaries,'armour','wald')
    
      if label == 'Historical L6':
        (likelihood,normalised_likelihood)=update_ecs_kasia(uprior,bin_centres,n_samples,bin_boundaries,'andrews','wald')
    
      if label == 'Palaeo Likelihood':
        (likelihood,normalised_likelihood)=update_ecs_palaeo1(uprior,bin_centres)
    
      if label == 'Palaeo L1':
        (likelihood,normalised_likelihood)=update_ecs_palaeo1(uprior,bin_centres)
    
      if label == 'Palaeo L2':
        (likelihood,normalised_likelihood)=update_ecs_palaeo2(uprior,bin_centres)
    
      if label == 'Process Likelihood':
        (likelihood,normalised_likelihood)=update_ecs_process_s_minus_2(uprior,bin_centres,n_samples,bin_boundaries,normal01_2co2)
        
      if label == 'Process L1':
        (likelihood,normalised_likelihood)=update_ecs_process1(uprior,bin_centres,n_samples,bin_boundaries,normal01_2co2)
        
      if label == 'Process L2':
        (likelihood,normalised_likelihood)=update_ecs_process2(uprior,bin_centres,n_samples,bin_boundaries,normal01_2co2)
    
      print ("label=",label)

      np.savetxt('bin_boundaries.txt',bin_boundaries)
      np.savetxt(label+'.normalised_likelihood.txt',normalised_likelihood)
      np.savetxt(label+'.likelihood.txt',likelihood)

  print ("label=",label)

  likelihood=np.power(likelihood,exponent)
  print 'exponent=',exponent
  assert (exponent <= 1 )
  assert (exponent >= 0 )
  normalised_likelihood=normalise_pdf(uprior*likelihood,bin_width)
  posterior = normalise_pdf(prior * likelihood,bin_width)
  
  return(likelihood,normalised_likelihood,posterior)

def setup_bins():
  # reducing this number biases the calculation of the probabilities
  n_samples=40000000L
  n_bins=100000L

  quick_and_dirty=0
  if quick_and_dirty:
    n_samples=1000000L
    n_bins=100000L

  bin_boundaries=np.linspace(-1000,1000,n_bins)
  bin_width=bin_boundaries[1]-bin_boundaries[0]
  bin_centres=bin_boundaries[0:n_bins-1]+bin_width/2
  return (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)
     
def plot_ecs_priors(fomu,fosigma,femu,fesigma):

  print 'fomu=',fomu
  print 'fosigma=',fosigma
  print 'femu=',femu
  print 'fesigma=',fesigma

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  color='k'
  xrange=[-1,15]
  yrange=[0,1]
  plt.xlim(xrange)
  plt.ylim(yrange)
  #plt.title(label)
  #plt.plot(xrange,[0.5,0.5],color=color)
  #plt.plot([1,1],yrange,color=color)
  plt.plot([1.5,1.5],yrange,color=color)
  plt.plot([4.5,4.5],yrange,color=color)

  xpos=5.0
  dypos=0.03
  ypos=0.8+dypos/2

  tickscale=0.5
  ticklen=0.05

  color='r'
  label1_1='PR1 P(S)=S^-2'
  label1_2=''
  prior1=calc_s_semi_inf_lambda_prior(n_samples)
  plt.plot(bin_centres,prior1*20,linewidth=2,color=color)
  ypos=ypos-dypos
  plt.text(xpos,ypos,label1_1,color=color)

  (ecs5,ecs50,ecs95)=ecs_cdf(prior1,bin_boundaries,bin_width,label1_1+' '+label1_2,label1_2)

#  plt.plot([ecs5,ecs5],[0,ticklen*tickscale],linewidth=2,color=color)
#  plt.plot([ecs95,ecs95],[0,ticklen*tickscale],linewidth=2,color=color)
#  plt.plot([ecs50,ecs50],[0,ticklen*tickscale*2],linewidth=2,color=color)

  color='g'
  label2_1='PR2 S~U(0,20)'
  label2_2=''
  prior2=calc_uniform_s_prior(0,20,n_samples)
  plt.plot(bin_centres,prior2,linewidth=2,color=color)
  ypos=ypos-dypos
  plt.text(xpos,ypos,label2_1,color=color)
    
  (ecs5,ecs50,ecs95)=ecs_cdf(prior2,bin_boundaries,bin_width,label2_1+' '+label2_2,label2_2)

#  plt.plot([ecs5,ecs5],[0,ticklen*tickscale],linewidth=2,color=color)
#  plt.plot([ecs95,ecs95],[0,ticklen*tickscale],linewidth=2,color=color)
#  plt.plot([ecs50,ecs50],[0,ticklen*tickscale*2],linewidth=2,color=color)

  if 0:
      color='r'
      label1_1='PR1 SSBW-like S=-F/L'
      label1_2='F ~ N(3.7,0.74) L ~ N(-1.8,0.9)'
      prior1=calc_ssbw_prior(-1.8,0.9,3.7,0.74,label1_2,n_samples)
      plt.plot(bin_centres,prior1,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_2,color=color)
    
      color='m'
      nsd=2
      label2_1='PR2 UFL-SSBW-like S=-F/L'
      label2_2='F~U(2.2,5.2) L~U(-3.6,0)'
      prior2=calc_uniform_f_lambda_prior(3.7-3.7*.2*nsd,3.7+3.7*.2*nsd,-1.8-0.9*nsd,-1.8+0.9*nsd,n_samples)
      plt.plot(bin_centres,prior2,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label2_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label2_2,color=color)
      
      color='c'
      label3_1='PR3 UFL S=-F/L'
      label3_2='F~U(0,10.8) L~U(-3.6,0)'
      prior3=calc_uniform_f_lambda_prior(0,10.8,-3.6,0,n_samples)
      plt.plot(bin_centres,prior3,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label3_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label3_2,color=color)
      
      color='k'
      label4_1='PR4 UFL S=-F/L'
      label4_2='F~U(0,18) L~U(-3.6,0)'
      prior4=calc_uniform_f_lambda_prior(0,18,-3.6,0,n_samples)
      plt.plot(bin_centres,prior4,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label4_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label4_2,color=color)
      
      color='g'
      label5_1='PR5 UFL S=-F/L'
      label5_2='F~U(0,36) L~U(-3.6,0)'
      prior5=calc_uniform_f_lambda_prior(0,36,-3.6,0,n_samples)
      plt.plot(bin_centres,prior5,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label5_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label5_2,color=color)
      
      color='r'
      label1_1='PR1.1 S=-F/L'
      label1_2='F ~ N(3.7,0.3) L ~ U(-2.6,0)'
      prior1=calc_normal_f_uniform_lambda_prior(3.7,0.3,-2.6,0,n_samples)
      plt.plot(bin_centres,prior1,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_2,color=color)
        
      color='g'
      label1_1='PR1.2 S=-F/L'
      label1_2='F ~ N(3.7,0.3) L ~ U(-10,0)'
      prior1=calc_normal_f_uniform_lambda_prior(3.7,0.3,-10,0,n_samples)
      plt.plot(bin_centres,prior1,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_2,color=color)
        
      color='b'
      label1_1='PR1.3 S=-F/L'
      label1_2='F ~ N(3.7,0.3) L ~ U(-30,0)'
      prior1=calc_normal_f_uniform_lambda_prior(3.7,0.3,-30,0,n_samples)
      plt.plot(bin_centres,prior1,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label1_2,color=color)
        
      color='m'
      label3_1='PR3 S=-F/L'
      label3_2='F~U(0,7.4 L~U(-2.6,0)'
      prior3=calc_uniform_f_lambda_prior(0,7.4,-2.6,0,n_samples)
      plt.plot(bin_centres,prior3,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label3_1,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label3_2,color=color)
          
      if 0:
          color='m'
          label4_1='PR4 S ~ U(0,5)'
          label4_2=''
          prior4=calc_uniform_s_prior(0,5,n_samples)
          plt.plot(bin_centres,prior4,linewidth=2,color=color)
          ypos=ypos-dypos
          plt.text(xpos,ypos,label4_1,color=color)
          label4_2='S ~ U(0,5)'
          
      color='y'
      label5_1='PR5 Cauchy Prior'
      label5_2='S ~ C(2.5,3)'
      cauchy_location=2.5
      cauchy_scale=3
      prior5=calc_cauchy_like_prior_ah09(cauchy_location,cauchy_scale,n_samples)
      plt.plot(bin_centres,prior5,linewidth=2,color=color)
      ypos=ypos-dypos
      plt.text(xpos,ypos,label5_1+' '+label5_2,color=color)
    
#  color='#ffa500' # orange
#  label6_1='PR6 Cauchy Prior'
#  label6_2='S ~ C(3,12)'
#  cauchy_location=3
#  cauchy_scale=12
#  prior6=calc_cauchy_like_prior_ah09(cauchy_location,cauchy_scale,n_samples)
#  plt.plot(bin_centres,prior6,linewidth=2,color=color)
#  ypos=ypos-dypos
#  plt.text(xpos,ypos,label6_1+' '+label6_2,color=color)

#  print label
#  print 'forcing mean   = ' , fomu
#  print 'forcing sigma  = ' , fosigma
#  print 'feedback mean  = ' , femu
#  print 'feedback sigma = ' , fesigma
#  print 'forcing range =' ,fos.min(),fos.max()
#  print 'feedback range = ',fes.min(),fes.max()
#  print 'range ecs = ',ecss.min(),ecss.max()

#  plt.text(xpos,0.85,'P(ECS < 1.5) = '+"{:.2f}".format(p_ecs_lt_1p5),color=color)
#  plt.text(xpos,0.8,'P(ECS > 4.5) = '+"{:.2f}".format(p_ecs_gt_4p5),color=color)
#  plt.text(xpos,0.75,'ECS 5th percentile = '+"{:.1f}".format(ecs5),color=color)
#  plt.text(xpos,0.7,'ECS 95th percentile = '+"{:.1f}".format(ecs95),color=color)

  plt.show()

def update_average_likelihoods(prior,labels,normal01_2co2):

  import os.path

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  # normalise likelihoods by combining them with a uniform prior over -100,100
  f='u110.txt'
  if os.path.isfile(f):
      uprior=np.loadtxt(f)
  else:
      uprior=calc_uniform_s_prior(-10,100,n_samples)
      np.savetxt(f,uprior)

  average_likelihood=bin_centres*0.0
  for i in range(len(labels)):
    (likelihood,normalised_likelihood,posterior)=likelihood_update(labels[i],normal01_2co2,uprior,prior,1)
    average_likelihood=average_likelihood+likelihood

  likelihood=normalise_pdf(average_likelihood,bin_width)
    
  posterior_numerator = prior * likelihood
  posterior=posterior_numerator/np.sum(posterior_numerator, dtype=np.float64)/bin_width

  return (likelihood,posterior)

def plot_ecs_pdf():
    
  import os.path

  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  f='u100.txt'
  if os.path.isfile(f):
      uprior=np.loadtxt(f)
  else:
      uprior=calc_uniform_s_prior(0,100,n_samples)
      np.savetxt(f,uprior)

  #90% confidence interval for normal distribution is +/-1.645 * sigma
  #80% confidence interval for normal distribution is 1+/-.28 * sigma 
    
  (n_bins,bin_boundaries,bin_centres,bin_width,n_samples)=setup_bins()

  brown='#654321' 
  orange='#ffa500' 
  grey='#808080' 
  colors=['k','r','g','b','m','c','y',orange,grey,brown] 

  po1='PO1 Baseline'
  po2='PO2 Uniform Prior P2'
  po3='PO3 No Process'
  po4='PO4 No Historical'
  po5='PO5 No Palaeo'
  po6='PO6 IE=0.7'
  po7='PO7 IE=0.5'
  po8='PO8 Pro/His Dep'
  po9='PO9 His/Pal Dep'
  po10='PO10 Pal/Pro Dep'

  posterior_labels=[po1,po2,po3,po4,po5,po6,po7,po8,po9,po10]

  for j in range(len(posterior_labels)):

      plabel=posterior_labels[j]

      # default settings PO1 XXXX
      inflation_exponent=1.0
      inflation_string=''
      select_prior='s-2'
      labels=['Process Likelihood','Historical Likelihood','Palaeo Likelihood']
      exponent=[1,1,1]
      s_exponent=['','','']
       
      if plabel == po2:
          select_prior='u20'
      if plabel == po3:
          select_prior='u20'
          labels=['Historical Likelihood','Palaeo Likelihood']
      if plabel == po4:
          labels=['Process Likelihood','Palaeo Likelihood']
      if plabel == po5:
          labels=['Process Likelihood','Historical Likelihood']
      if plabel == po6:
          inflation_exponent=0.7
          inflation_string='^0.7'
      if plabel == po7:
          inflation_exponent=0.5
          inflation_string='^0.5'
      if plabel == po8:
          exponent=[0.5,0.5,1]
          s_exponent=['0.5','0.5','']
      if plabel == po9:
          exponent=[1,0.5,0.5]
          s_exponent=['','0.5','0.5']
      if plabel == po10:
          exponent=[0.5,1,0.5]
          s_exponent=['0.5','','0.5']
    
      color=colors[j]

      # calculate gaussian samples to correlate CO2 forcings
    
      ticklen=0.05
      
      xrange=[0,10]
      #xrange=[-1,300]
      yrange=[0,0.7]
      plt.xlim(xrange)
      plt.ylim(yrange)
      plt.plot([1.5,1.5],yrange,color='k')
      plt.plot([4.5,4.5],yrange,color='k')
      
      # main prior 

      #select_prior='u100'
      #select_prior='u10'
      #select_prior='u5'
      #select_prior='cauchy25'
      #select_prior='cauchy3'
      #select_prior='process_pdf'
      #select_prior='ssbw'
      #select_prior='u20'
    


      #prior=calc_uniform_f_lambda_prior(0,100,-10,0,n_samples)
      #prior_label='UFL Prior'
      
      #prior=calc_uniform_s_prior(0,20,n_samples)
      #prior_label='SU Prior S~U(0,20)'
    
      #prior=calc_uniform_s_prior(0,40,n_samples)
      #label1='SU Prior S~U(0,40)'
      #label2=''
    
      #prior=calc_uniform_s_prior(0,10,n_samples)
      #prior_label='SU Prior S~U(0,10)'
    
      if select_prior == 'process_pdf':
          f='normal01_2co2.txt'
          if os.path.isfile(f):
              normal01_2co2=np.load(f)
          else:
              normal01_2co2=np.random.normal(0,1,n_samples) 
              np.save(f,normal01_2co2)
    
          (likelihood,prior)=update_ecs_process1(uprior,bin_centres,n_samples,bin_boundaries,normal01_2co2)
          label1='Process PDF Prior'
          label2=''
    
      if select_prior == 's-2':
          prior=calc_s_semi_inf_lambda_prior(n_samples)
          label1='S^-2 Prior'
          label2=''
    
      if select_prior == 'u100':
          prior=calc_uniform_s_prior(0,100,n_samples)
          label1='SU Prior S~U(0,100)'
          label2=''
    
      if select_prior == 'u40':
          prior=calc_uniform_s_prior(0,40,n_samples)
          label1='SU Prior S~U(0,40)'
          label2=''
    
      if select_prior == 'u20':
          prior=calc_uniform_s_prior(0,20,n_samples)
          label1='SU Prior S~U(0,20)'
          label2=''
    
      if select_prior == 'u10':
          prior=calc_uniform_s_prior(0,10,n_samples)
          label1='SU Prior S~U(0,10)'
          label2=''
    
      if select_prior == 'u5':
          prior=calc_uniform_s_prior(0,5,n_samples)
          label1='SU Prior S~U(0,5)'
          label2=''
    
      if select_prior == 'ssbw':
          prior_label='SSBW-like Prior'
          prior=calc_ssbw_prior(-1.8,0.9,3.7,0.74,prior_label,n_samples)
          label1=prior_label
          label2=''
        
      if select_prior == 'cauchy25':
          cauchy_location=2.5
          cauchy_scale=3
          clabel='Cauchy-like Prior S~C(2.5,3)' 
          prior=calc_cauchy_like_prior_ah09(cauchy_location,cauchy_scale,n_samples)
          label1=clabel
          label2=''
        
      if select_prior == 'cauchy3':
          cauchy_location=3
          cauchy_scale=12
          clabel='Cauchy-like Prior S~C(3,12)' 
          prior=calc_cauchy_like_prior_ah09(cauchy_location,cauchy_scale,n_samples)
          label1=clabel
          label2=''
    
      #prior_label=clabel
    
      nsd=2
    
      #label1='UFL-SSBW-like S=-F/L'
      #label2='F~U(2.2,5.2) L~U(-3.6,0)'
      #prior=calc_uniform_f_lambda_prior(3.7-3.7*.2*nsd,3.7+3.7*.2*nsd,-1.8-0.9*nsd,-1.8+0.9*nsd,n_samples)
      
      #label1='UFL S=-F/L'
      #label2='F~U(0,10.8) L~U(-3.6,0)'
      #prior=calc_uniform_f_lambda_prior(0,10.8,-3.6,0,n_samples)
     
      # posterior 
      #label1='UFL S=-F/L'
      #label2='F~U(0,18) L~U(-3.6,0)'
      #prior=calc_uniform_f_lambda_prior(0,18,-3.6,0,n_samples)
      
      # posterior 10
      #label1='UFL S=-F/L'
      #label2='F~U(0,36) L~U(-3.6,0)'
      #prior=calc_uniform_f_lambda_prior(0,36,-3.6,0,n_samples)
    
      prior_label=label1+' '+label2
    
      update=np.copy(prior)
    
      normal01_2co2='now accessed from saved file at lower level'
    
      for i in range(len(labels)):
        (likelihood,normalised_likelihood,posterior)=likelihood_update(labels[i],normal01_2co2,uprior,update,exponent[i])
      #  plot_likelihood(labels[i],colors[i],i-4,1,normal01_2co2,uprior,exponent[i],s_exponent[i])
        update=np.copy(posterior)
    
      xpos=6
      
      # apply bayesian hierachical model
      #modes=np.array([palaeo_mode,hist_mode,process_mode])
      #mode_mean=np.mean(modes)
      #mode_stdev=np.std(modes)
      
      #print ("modes=",modes)
      #print ("mean=",mode_mean)
      #print ("stdev=",mode_stdev)
      
      #color='r'
      #print 'statistics for prior:'
      
      #plt.plot(bin_centres,prior,linewidth=2,color=color)
      #plt.text(xpos,0.9,prior_label,color=color)
      #label2=""
    
      # plot ah09 cauchy-like prior density using formula from annan and hargreaves 2009
      #cauchy_density=1.0/((bin_centres-cauchy_location)**2+cauchy_scale)
      
      # plot cauchy prior density using formula from https://itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
      #cauchy_density=1.0/(cauchy_scale*3.142*(1+((bin_centres-cauchy_location)/cauchy_scale)**2))
    
      #cauchy_density=cauchy_density/cauchy_density.max()*prior.max()
      #plt.plot(bin_centres, cauchy_density,color='m')
      
      #([p_ecs_lt_1,p_ecs_lt_1p5,p_ecs_gt_4p5,ecs5,ecs10,ecs17,ecs20,ecs50,ecs80,ecs83,ecs90,ecs95,ecs_mean,ecs_mode])=ecs_cdf(prior,bin_boundaries,bin_width,prior_label,label2)
      
      #plt.text(xpos,0.85,'P(ECS < 1.5) = '+"{:.2f}".format(p_ecs_lt_1p5),color=color)
      #plt.text(xpos,0.8,'P(ECS > 4.5) = '+"{:.2f}".format(p_ecs_gt_4p5),color=color)
      #plt.text(xpos,0.75,'ECS 5th percentile = '+"{:.1f}".format(ecs5),color=color)
      #plt.text(xpos,0.7,'ECS 95th percentile = '+"{:.1f}".format(ecs95),color=color)
      
      #print 'statistics for posterior:'
      
      posterior=normalise_pdf(posterior**inflation_exponent,bin_width)
    
      plt.plot(bin_centres,posterior,linewidth=2,color=color)
      plt.text(xpos,0.65-0.04*j,plabel,color=color)
      
      (ecs5,ecs50,ecs95)=ecs_cdf(posterior,bin_boundaries,bin_width,plabel,'')
      
      plot_posterior_stats=0
    
      if plot_posterior_stats:
          plt.text(xpos,0.45,'P(ECS < 1.5) = '+"{:.2f}".format(p_ecs_lt_1p5),color=color)
          plt.text(xpos,0.4,'P(ECS > 4.5) = '+"{:.2f}".format(p_ecs_gt_4p5),color=color)
          plt.text(xpos,0.35,'ECS 5th percentile = '+"{:.1f}".format(ecs5),color=color)
          plt.text(xpos,0.3,'ECS 95th percentile = '+"{:.1f}".format(ecs95),color=color)
          
      tickscale=0.5
      plt.plot([ecs5,ecs5],[0,ticklen*tickscale],linewidth=2,color=color)
      plt.plot([ecs95,ecs95],[0,ticklen*tickscale],linewidth=2,color=color)
      plt.plot([ecs50,ecs50],[0,ticklen*tickscale*2],linewidth=2,color=color)

  plt.show()
  
# main program

plot_ecs_pdf()
#plot_likelihoods()
#plot_ecs_priors(3.7,3.7*0.2,-1.8,0.9)

