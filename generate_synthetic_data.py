import pandas
import numpy
from math import exp
from scipy.stats import uniform, norm, beta
import seaborn
from scipy.optimize import minimize, fsolve
from scipy import integrate
import functools
import defopt


def generate_data(N: int=100, TF: int=200, LAMBDA0: float=0.005, ALPHA: float=0.7, TAU: int=100, BETA: float=0.0, TE: int=100, SIGMAE: int=50):
  """
  Generate synthetic data
  :param N: population size
  :param TF: follow up period (no of days)
  :param LAMBDA0: background hazard
  :param ALPHA: vaccine efficacy
  :param TAU: vaccine efficacy time
  :param BETA: relative increased risk of infection during epidemic
  :param TE: peak time of epidemic (no of days)
  :param SIGMAE: width of the epidemic bump (no of days)
  """

  # follow-up times (days) as an n x tf array
  ons = numpy.ones((N, 1), dtype=int)
  tt = numpy.matmul( ons, numpy.array( range(1, TF + 1) ).reshape((1, TF)) ) # N x TF array

  # vaccine age at start is a uniformly distributed random variable
  time_of_vaccination_at_start = 2*TAU * uniform.rvs(size=N, random_state=123) ## changed from 3!
  ons = numpy.ones((1, TF), dtype=int)
  ttv = numpy.matmul(time_of_vaccination_at_start.reshape((N, 1)), ons) # N x TF array

  infection = 1.0 + BETA*numpy.exp(-(tt - TE)**2/SIGMAE**2)
  lambda_ij = infection * LAMBDA0 * (1.0 - ALPHA*numpy.exp(-(tt + ttv)/TAU))

  # survival (not infected) prob in time, nodal
  survival = numpy.ones((N, TF + 1), numpy.float64)
  survival[:, 1:] = numpy.exp(-numpy.cumsum(lambda_ij, axis=1))

  # decide when individuals get infected. Each individual gets a prob (badluck) assigned. 
  # Infection occurs when badluck >= survival
  badluck = uniform.rvs(size=N, random_state=234)
  # badluck = numpy.minimum(0.999, norm.rvs(size=N, loc=0.5, scale=0.1, random_state=234))
  # scale = 0.5
  # mid = 0.5 # centred on 0.5
  # loc = mid - scale/2 # for the beta dist loc is the low end 
  # badluck = beta.rvs(a=2, b=2, loc=loc, scale=scale, size=N, random_state=234)

  ons = numpy.ones((1, TF + 1), dtype=int)
  badluck2d = numpy.matmul(badluck.reshape((N, 1)), ons) # N x (TF + 1) array
  not_infected_mask = (survival > badluck2d)
  #not_infected_mask = (survival > 0.5)


  # infection times
  time_to_infection = numpy.minimum(not_infected_mask.sum(axis=1), TF)
  status = numpy.array(((1 - not_infected_mask).sum(axis=1) > 0), dtype=int) # censored or infected

  data = pandas.DataFrame({'time_of_event': time_to_infection,
                          'time_of_vaccination_at_start': time_of_vaccination_at_start,
                          'status': status,
                          'not_infected_prob': survival[:, -1],
                          'badluck': badluck})

  data.to_csv('survival.csv')

if __name__ == '__main__':
  defopt.run(generate_data)