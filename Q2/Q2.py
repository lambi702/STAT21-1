from typing import List
from scipy.stats import beta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.special as sc
import scipy.optimize as sco

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


data = population(pd.read_csv("data.csv", index_col=0),[20191230,20190931])

def Q2 ():
    pop = np.random.choice(data.iloc[:,0],50)
    var = np.var(pop)
    mean = np.mean(pop)
    aMom = mean*((((1-mean)*mean)/var)-1)
    bMom = ((((1-mean)*mean)/(var))-1)*(1-mean)
    print("alpha MOM:", aMom)
    print("beta MOM:", bMom)

    x = sco.minimize(beta_log_likelihood,np.array([1,1]),pop)
    aMle = x.x[0]
    bMle = x.x[1]
    print("alpha MLE:", aMle)
    print("beta MLE: ", bMle)

    plt.figure()
    plt.hist(pop)
    plt.plot(beta(aMle,bMle))
    plt.show()
Q2()
