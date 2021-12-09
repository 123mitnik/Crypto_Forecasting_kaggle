'''
Data: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map
Reference: https://www.jmir.org/2021/2/e26081/#ref29
'''

import pandas as pd
from datetime import datetime
import numpy as np


df_case = pd.read_csv('covid_confirmed_usafacts.csv')
df_case = df_case[df_case['countyFIPS'] != 0]
population = pd.read_csv('covid_county_population_usafacts.csv')
population = population[population['countyFIPS'] != 0].sort_values(by = 'population',ascending= False)

# select cross-sectional
def section_id(n, df_pop, method):
	pop = df_pop.sort_values(by='population', ascending=False)
	pop = pop.set_index(['countyFIPS', 'State'])  # set multiple index
	if method == 'population':
		## top population county
		section_id = pop.iloc[range(n), :].index
	elif method == 'random':
		## random pick county
		section_id = pop.sample(n=n, random_state= 1).index
	elif method == 'state':
		pass
	return section_id
sections = section_id(n=200, df_pop=population, method = 'population')
df = df_case.set_index(['countyFIPS','State'])# set multiple index
df = df.loc[sections]


# select period
df = df.iloc[:, 2:]# remove index columns
def select_date(df , fr: str ='2021-09-01', to: str ='2021-09-01'):
	df.columns = pd.to_datetime(df.columns)
	format = "%Y-%m-%d"
	try:
		fr=datetime.strptime(fr, format)
		to=datetime.strptime(to, format)
	except ValueError:
		print("This is the incorrect date string format. It should be a string like 2000-01-25")
	dates = df.columns[[fr<=i<=to for i in df.columns]]
	df = df[dates]
	return df
df = select_date(df = df, fr = "2020-10-01", to = '2020-11-24')# with datetime index in the first row

## divide by population
pop = population.set_index(['countyFIPS', 'State']).loc[sections]
df = df.div(pop.iloc[:,1], axis='rows')

## by percentage change
df = df.T.resample('D').ffill().pct_change().T#daily/weekly percentage change
#df = df.T.pct_change(1).T#daily change
df = df.fillna(0)

# persistence CI
from paperClass import covidCI
from scipy.stats import norm
significance = 0.05
cv = norm.ppf(1 - significance / 2, loc=0, scale=1)

g = covidCI(df = df.to_numpy())
g(thetaci = 0, showmle = True)

grid = np.linspace(0, 1, 1000)
#ci = [g(thetaci=th, showmle = False) for th in grid]
#ci_id = [np.absolute(c)<=cv for c in ci]
ci_id = [np.absolute(g(thetaci=th, showmle = False))<=cv for th in grid]

'''
tstat_ci = grid[[id[0] for id in ci_id]]
HKstable_ci = grid[[id[1] for id in ci_id]]
HKnonstable_ci = grid[[id[2] for id in ci_id]]
M_ci = grid[[id[3] for id in ci_id]]
result = [tstat_ci,HKstable_ci,HKnonstable_ci,M_ci]
'''
result = [grid[[id[i] for id in ci_id]] for i in range(4)]
name = ['tstat_ci','HKstable_ci','HKnonstable_ci','M_ci']
minmax = [(c.min(),c.max()) if len(c)>0 else None for c in result]
[print(f'{name[i]} = {minmax[i]}') for i in range(4)]


