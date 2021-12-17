from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.special as sc

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

#load data from the csv
data = pd.read_csv("data.csv",index_col=0)

#print Our personal data for the FIXED_COUNTRIES (Q1.a)

def Q1 (data):
    #Q1.a
    pop = population(data, [20191230, 20190931])
    print(pop[146:][:])
    print("")
    
    #Q1.b
    mean = np.mean(data, axis = 0)
    print("Moyenne : ")
    print(mean)
    print("")

    print("Standard deviation : ")
    std = np.std(data, axis = 0)
    print(std)
    print("")

    median = np.median(data, axis=0)
    print("Median :", median)
    print("")

    quantile1 = np.quantile(data, 1/4, axis=0)
    print("Q1: ",quantile1)
    quantile2 = np.quantile(data, 3/4, axis=0)
    print("Q2 :", quantile2)

    ay1 = plt.subplot(131)
    ay1.axes.get_xaxis().set_visible(False)
    plt.boxplot(data.iloc[:,0])
    plt.title("TOP 10 % richer")

    ay2 = plt.subplot(132)
    ay2.axes.get_xaxis().set_visible(False)
    plt.boxplot(data.iloc[:, 1])
    plt.title("CO2 / Habitant (in T)")

    ay3 = plt.subplot(133)
    ay3.axes.get_xaxis().set_visible(False)
    plt.boxplot(data.iloc[:, 2])
    plt.title("PIB / Habitan")
    plt.tick_params('y',labelleft = False, labelright = True, right = True, left = False, bottom = False)
    plt.show()




Q1(data)