import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.special as sc

from typing import List

FIXED_COUNTRIES = ["USA", "Belgium", "China", "Togo"]


def population(data: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    """
    Extract a population for the original dataset.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataset obtained with pandas.read_csv
    ids: List[int]
        List of ULiege ids for each group member (e.g. s167432 and s189134 -> [20167432,20189134])

    Returns
    -------
    DataFrame containing your population
    """
    pop = data.drop(FIXED_COUNTRIES).sample(146, random_state=sum(ids))
    for c in FIXED_COUNTRIES:
        pop.loc[c] = data.loc[c]
    return pop


def beta_log_likelihood(theta, *x):
    """
    Function equal to -log L(\theta;x) to be fed to scipy.optimize.minimize

    Parameters
    ----------
    theta: theta[0] is alpha and theta[1] is beta
    x: x[0] is the data
    """
    a = theta[0]
    b = theta[1]
    n = len(x[0])

    # Log-likelihood
    obj = (a - 1) * np.log(x[0]).sum() + (b - 1) * np.log(1 - x[0]).sum() - n * np.log(sc.beta(a, b))
    # We want to maximize
    sense = -1

    return sense * obj


def scientific_delta(pop: pd.DataFrame) -> float:
    """

    Parameters
    ----------
    pop: pandas.DataFrame
        Dataframe containing a column 'PIB_habitant' and 'CO2_habitant'

    Returns
    -------
    Delta value computed by scientists
    """
    median_gdp = pop["PIB_habitant"].median()
    pop["Rich"] = pop.apply(lambda x: x["PIB_habitant"] >= median_gdp, axis=1)
    means = pop.groupby("Rich")['CO2_habitant'].mean()
    return means[True] - means[False]

def compute_pivot(var: pd.DataFrame, pop_size: int, confidence: np.float64):
    mu = var.sample(n=pop_size).mean()
    double_size = pop_size*2
    chi2_compute = [chi2.ppf(1 - confidence/2, double_size), chi2.ppf(confidence/2, double_size)]

    return [1/(2*pop_size*mu*(1/chi2_compute[1])), 1/(2*pop_size*mu*(1/chi2_compute[0]))]

def compute_bootstrap(var: pd.DataFrame, pop_size: int, bootstrap_size: int, confidence: np.float64):
    conf_level = 1 - confidence
    var_np = var.sample(n=pop_size).to_numpy()
    compute = bootstrap((var_np,), np, np.std, confidence_level=conf_level, n_resamples=bootstrap_size)

    conf_inter0 = compute.confidence_interval[0]
    conf_inter1 = compute.confidence_interval[1]

    return [1/conf_inter1, 1/conf_inter0]

def Q3(pop: pd.DataFrame, pop_size: int, samples: int, bootstrap_size: int):
    confidence  = np.float64(0.05)
    given_lambda = 5.247 * 10 ** (-5)
    var = pop.loc[:, 'PIB_habitant']
    range_5inc = range(5,pop_size+1,5)

    pivot_size_steps = np.zeros(len(range_5inc))
    bootstrap_size_steps = np.zeros(len(range_5inc))

    pivot_count_steps = np.zeros(len(range_5inc))
    bootstrap_count_steps = np.zeros(len(range_5inc))
    
    size_pivot = np.zeros(samples)
    size_bootstrap = np.zeros(samples)

    print('pivot :', compute_pivot(var,pop_size,confidence))
    print('bootstrap :', compute_bootstrap(var,pop_size,bootstrap_size,confidence))

    for i in range_5inc:
        count_pivot = 0
        count_bootstrap = 0
        for j in range(samples):
            
            computed_pivot = compute_pivot(var,i,confidence)
            if(computed_pivot[0] < given_lambda < computed_pivot[1]):
                count_pivot = count_pivot + 1
            
            computed_bootstrap = compute_bootstrap(var,i,bootstrap_size,confidence)
            if computed_bootstrap[0] < given_lambda < computed_pivot[1]:
                count_bootstrap = count_bootstrap + 1
            
            size_pivot[j] = computed_pivot[1] - computed_pivot[0]
            size_bootstrap[i] = computed_bootstrap[1] - computed_bootstrap[0]

        j = range_5inc.index(i)
        pivot_size_steps[j] = size_pivot.mean()
        pivot_count_steps[j] = count_pivot/samples

        bootstrap_size_steps[j] = size_bootstrap.mean()
        bootstrap_count_steps[j] = count_bootstrap/samples

    plt.plot(range_5inc, pivot_size_steps)
    plt.plot(range_5inc,bootstrap_size_steps)
    plt.lengend(['pivot', 'bootstrap'])
    plt.show()
    
    plt.plot(range_5inc,pivot_count_steps)
    plt.plot(range_5inc,bootstrap_count_steps)
    plt.lengend(['pivot', 'bootstrap'])
    plt.show()

data = pd.read_csv("data.csv", index_col=0)
ids = [20190931, 20191230]
population(data,ids)

Q3(population,50,500,100)