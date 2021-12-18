import numpy as np
import pandas as pd

from scipy.stats import beta, chi2, bootstrap, t
import scipy.optimize as sco

import matplotlib.pyplot as plt

import scipy.special as sc

from typing import List

FIXED_COUNTRIES = ["USA", "Belgium", "China", "Togo"]


#############################
# Functions from helpers.py #
#############################

def population(data_csv: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    """
    Extract a population for the original dataset.

    Parameters
    ----------
    data_csv: pandas.DataFrame
        Dataset obtained with pandas.read_csv
    ids: List[int]
        List of ULiege ids for each group member (e.g. s167432 and s189134 -> [20167432,20189134])

    Returns
    -------
    DataFrame containing your population
    """
    pop = data_csv.drop(FIXED_COUNTRIES).sample(146, random_state=sum(ids))
    for c in FIXED_COUNTRIES:
        pop.loc[c] = data_csv.loc[c]
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


###########################
# Complementary functions #
###########################


def compute_pivot(var: pd.DataFrame, pop_size: int, confidence: np.float64):
    mu = var.sample(n=pop_size).mean()
    double_size = pop_size * 2
    chi2_compute = [chi2.ppf(1 - confidence / 2, double_size), chi2.ppf(confidence / 2, double_size)]

    return [1 / (2 * pop_size * mu * (1 / chi2_compute[1])), 1 / (2 * pop_size * mu * (1 / chi2_compute[0]))]


def compute_bootstrap(var: pd.DataFrame, pop_size: int, bootstrap_size: int, confidence: np.float64):
    conf_level = 1 - confidence
    var_np = var.sample(n=pop_size).to_numpy()
    compute = bootstrap((var_np,), np.std, confidence_level=conf_level, n_resamples=bootstrap_size)

    conf_inter0 = compute.confidence_interval[0]
    conf_inter1 = compute.confidence_interval[1]

    return [1 / conf_inter1, 1 / conf_inter0]


def test_hypothesis(pop: pd.DataFrame, samples: int, alpha: np.float64, delta: np.float64, passengers: int):
    percentage = 0.0
    for i in range(samples):
        testPop = pop.sample(n=passengers)
        PIB = testPop.loc[:, 'PIB_habitant']
        CO2 = testPop.loc[:, 'CO2_habitant']

        median_PIB = PIB.median()

        CO2OfCountriesAboveMedianPIB = CO2[PIB > median_PIB]
        CO2OfCountriesUnderMedianPIB = CO2[PIB <= median_PIB]
        deltaPop = CO2OfCountriesAboveMedianPIB.mean() - CO2OfCountriesUnderMedianPIB.mean()

        size_above = len(CO2OfCountriesAboveMedianPIB)
        size_under = len(CO2OfCountriesUnderMedianPIB)
        inv = (1 / size_above) + (1 / size_under)
        degreesOfFreedom = size_above + size_under - 2
        variance = CO2OfCountriesAboveMedianPIB.var() + CO2OfCountriesUnderMedianPIB.var()

        variance_dof = 1 / degreesOfFreedom * variance

        cond = delta + t.ppf((1 - alpha), degreesOfFreedom) * np.sqrt(variance_dof) * np.sqrt(inv)

        if cond < deltaPop:
            percentage = percentage + 1

    return percentage / samples


############################
# Q1 : Analyse descriptive #
############################


def Q1(data_csv: pd.DataFrame, save: bool):
    print("\n ### Q1 ###\n")
    pop = population(data_csv, [20191230, 20190931])
    print(pop[146:][:])
    print("")

    mean = np.mean(data_csv, axis=0)
    print("Moyenne : ")
    print(mean)
    print("")

    print("Standard deviation : ")
    std = np.std(data_csv, axis=0)
    print(std)
    print("")

    median = np.median(data_csv, axis=0)
    print("Median :", median)
    print("")

    quantile1 = np.quantile(data_csv, 1 / 4, axis=0)
    print("Quart1: ", quantile1)
    quantile2 = np.quantile(data_csv, 3 / 4, axis=0)
    print("Quart2 :", quantile2)

    bminTOP10 = quantile1[0] - (1.5 * (quantile2[0] - quantile1[0]))
    bmaxTOP10 = quantile2[0] + (1.5 * (quantile2[0] - quantile1[0]))
    print("min TOP10 : ", bminTOP10)
    print("max TOP10 : ", bmaxTOP10)

    bminCO2 = quantile1[1] - (1.5 * (quantile2[1] - quantile1[1]))
    bmaxCO2 = quantile2[1] + (1.5 * (quantile2[1] - quantile1[1]))
    print("min CO2 : ", bminCO2)
    print("max CO2 : ", bmaxCO2)

    bminPIB = quantile1[2] - (1.5 * (quantile2[2] - quantile1[2]))
    bmaxPIB = quantile2[2] + (1.5 * (quantile2[2] - quantile1[2]))
    print("min PIB : ", bminPIB)
    print("max PIB : ", bmaxPIB)

    ay1 = plt.subplot(131)
    ay1.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 0])
    plt.title("10% richest \nPIB proportion")

    ay2 = plt.subplot(132)
    ay2.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 1])
    plt.title("CO2 / Habitant (in T)")

    ay3 = plt.subplot(133)
    ay3.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 2])
    plt.title("PIB / Habitant")
    plt.tick_params('y', labelleft=False, labelright=True,
                    right=True, left=False, bottom=False)
    if save:
        plt.savefig("figs/boxplot.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 0])
    plt.title("Histogram of 10% richest PIB proportion")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_TOP10.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 1])
    plt.title("Histogram of CO2/Habitant")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_CO2.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 2])
    plt.title("Histogram of PIB/Habitant")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_PIB.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 0]), np.linspace(
        0, 1, len(data_csv.iloc[:, 0]), endpoint=False))
    plt.title("ECDF 10% richest PIB proportion")
    if save:
        plt.savefig("figs/ECDF_TOP10.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 1]), np.linspace(
        0, 1, len(data_csv.iloc[:, 1]), endpoint=False))
    plt.title("figs/ECDF of CO2/Habitant")
    if save:
        plt.savefig("figs/ECDF_CO2.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 2]), np.linspace(
        0, 1, len(data_csv.iloc[:, 2]), endpoint=False))
    plt.title("ECDF of PIB/Habitant")
    if save:
        plt.savefig("figs/ECDF_PIB.svg")
    plt.show()

    plt.subplot(231)
    plt.scatter(data_csv.iloc[:, 0], data_csv.iloc[:, 1], s=1)
    plt.title("TOP 10")
    plt.subplot(232)
    plt.text(0.45, 0.45, s="CO2", fontsize="x-large")
    plt.axis("off")
    plt.subplot(234)
    plt.scatter(data_csv.iloc[:, 0], data_csv.iloc[:, 2], s=1)
    plt.subplot(235)
    plt.scatter(data_csv.iloc[:, 1], data_csv.iloc[:, 2], s=1)
    plt.tick_params('y', labelleft=False)
    plt.subplot(236)
    plt.text(0, 0.45, s="PIB", fontsize="x-large")
    plt.axis("off")
    plt.suptitle("Matrix plot                              ")
    if save:
        plt.savefig("figs/matrix_plot.svg")
    plt.show()


##############################
# Q2 : Estimation ponctuelle #
##############################


def Q2(data_csv: pd.DataFrame, save: bool):
    print("\n ### Q2 ###\n")
    pop = np.random.choice(data_csv.iloc[:, 0], 50)
    var = np.var(pop)
    mean = np.mean(pop)
    aMom = mean * ((((1 - mean) * mean) / var) - 1)
    bMom = ((((1 - mean) * mean) / var) - 1) * (1 - mean)
    print("alpha MOM:", aMom)
    print("beta MOM:", bMom)

    (aMle, bMle) = Mle(pop)
    print("alpha MLE:", aMle)
    print("beta MLE: ", bMle)

    plt.figure()

    plt.hist(pop, density=True)
    x = np.arange(0.01, 1, 0.01)
    y = (beta.pdf(x, aMle, bMle))
    plt.plot(x, y, label='MLE')
    y = beta.pdf(x, aMom, bMom)
    plt.plot(x, y, label='MOM')
    plt.title("Real data compared to Beta distribution \nwith estimated parameters")
    plt.legend()
    if save:
        plt.savefig("figs/MLE_MOM_Beta.svg")
    plt.show()

    test = [50, 20, 40, 60, 80, 100]
    for j in test:
        echantillon1 = np.zeros((500, 2))
        echantillon2 = np.zeros((500, 2))
        for i in range(500):
            pop = np.random.choice(data_csv.iloc[:, 0], j)
            echantillon1[i] = Mom(pop)
            echantillon2[i] = Mle(pop)

        print(j, ": Biais aMom: ", echantillon1[:, 0].mean() - 13.35)
        print(j, ": Biais bMom: ", echantillon1[:, 1].mean() - 16.31)

        print(j, ": Variance aMom: ", np.var(echantillon1[:, 0]))
        print(j, ": Variance bMom: ", np.var(echantillon1[:, 1]))

        print(j, ": Quad Error aMom: ", np.square(
            np.subtract(echantillon1[:, 0], 13.35)).mean())
        print(j, ": Quad Error bMom: ", np.square(
            np.subtract(echantillon1[:, 1], 16.31)).mean())

        print(j, ": Biais aMle: ", echantillon2[:, 0].mean() - 13.35)
        print(j, ": Biais bMle: ", echantillon2[:, 1].mean() - 16.31)

        print(j, ": Variance aMle: ", np.var(echantillon2[:, 0]))
        print(j, ": Variance bMle: ", np.var(echantillon2[:, 1]))

        print(j, ": Quad Error aMle: ", np.square(
            np.subtract(echantillon2[:, 0], 13.35)).mean())
        print(j, ": Quad Error bMle: ", np.square(
            np.subtract(echantillon2[:, 1], 16.31)).mean())
        if j == 50:
            print("\n --- Bonus ---")


def Mle(pop: pd.DataFrame):
    x = sco.minimize(beta_log_likelihood, np.array([1, 1]), pop)
    aMle = x.x[0]
    bMle = x.x[1]
    return aMle, bMle


def Mom(pop: pd.DataFrame):
    var = np.var(pop)
    mean = np.mean(pop)
    aMom = mean * ((((1 - mean) * mean) / var) - 1)
    bMom = ((((1 - mean) * mean) / var) - 1) * (1 - mean)
    return aMom, bMom


##################################
# Q3 : Estimation par intervalle #
##################################

def Q3(pop: pd.DataFrame, pop_size: int, samples: int, bootstrap_size: int, save: bool):
    print("\n ### Q3 ###\n")
    confidence = np.float64(0.05)
    given_lambda = 5.247 * 10 ** (-5)
    var = pop.loc[:, 'PIB_habitant']
    range_5inc = range(5, pop_size + 1, 5)

    pivot_size_steps = np.zeros(len(range_5inc))
    bootstrap_size_steps = np.zeros(len(range_5inc))

    pivot_count_steps = np.zeros(len(range_5inc))
    bootstrap_count_steps = np.zeros(len(range_5inc))

    size_pivot = np.zeros(samples)
    size_bootstrap = np.zeros(samples)

    print('pivot :', compute_pivot(var, pop_size, confidence))
    print('bootstrap :', compute_bootstrap(var, pop_size, bootstrap_size, confidence))

    for i in range_5inc:
        count_pivot = 0
        count_bootstrap = 0
        for j in range(samples):

            computed_pivot = compute_pivot(var, i, confidence)
            if computed_pivot[0] < given_lambda < computed_pivot[1]:
                count_pivot = count_pivot + 1

            computed_bootstrap = compute_bootstrap(var, i, bootstrap_size, confidence)
            if computed_bootstrap[0] < given_lambda < computed_pivot[1]:
                count_bootstrap = count_bootstrap + 1

            size_pivot[j] = computed_pivot[1] - computed_pivot[0]
            size_bootstrap[i] = computed_bootstrap[1] - computed_bootstrap[0]

        j = range_5inc.index(i)
        pivot_size_steps[j] = size_pivot.mean()
        pivot_count_steps[j] = count_pivot / samples

        bootstrap_size_steps[j] = size_bootstrap.mean()
        bootstrap_count_steps[j] = count_bootstrap / samples

    plt.plot(range_5inc, pivot_size_steps)
    plt.plot(range_5inc, bootstrap_size_steps)
    plt.title('Evolution de la taille moyenne de l intervalle\nen fonction de la taille de l echantillon')
    plt.xlabel('Taille de l echantillon')
    plt.ylabel('Taille de l intervalle')
    plt.legend(['Pivot', 'Bootstrap'])
    if save:
        plt.savefig("figs/intervalle_echantillon.svg")
    plt.show()

    plt.plot(range_5inc, pivot_count_steps)
    plt.plot(range_5inc, bootstrap_count_steps)
    plt.title(
        'Evolution de la proportion d intervalles contenant la vraie valeur lambda\nen fonction de la taille de l '
        'echantillon')
    plt.xlabel('Taille de l echantillon')
    plt.ylabel('Proportion d intervalles contenant vrai lambda')
    plt.legend(['Pivot', 'Bootstrap'])
    if save:
        plt.savefig("figs/prop_lambda.svg")
    plt.show()


#########################
# Q4 : Test d'hypothèse #
#########################


def Q4(pop: pd.DataFrame, alpha: np.float64, samples: int):
    print("\n ### Q4 ###\n")
    PIB = pop.loc[:, 'PIB_habitant']
    CO2 = pop.loc[:, 'CO2_habitant']

    median_PIB = PIB.median()

    CO2OfCountriesAboveMedianPIB = CO2[PIB > median_PIB]
    CO2OfCountriesUnderMedianPIB = CO2[PIB <= median_PIB]

    deltaPop = CO2OfCountriesAboveMedianPIB.mean() - CO2OfCountriesUnderMedianPIB.mean()
    scDelta = scientific_delta(pop)

    print('Error : ', np.abs(deltaPop - scDelta))
    print('test 75 : ', test_hypothesis(pop, samples, alpha, deltaPop, 75))
    print('test 25 : ', test_hypothesis(pop, samples, alpha, deltaPop, 25))


###############
# Main script #
###############

data = pd.read_csv("data.csv", index_col=0)
studentsID = [20190931, 20191230]
populationTest = population(data, studentsID)

Q1(data, True)
Q2(data, True)
Q3(populationTest, 50, 500, 100, True)
Q4(populationTest, 0.05, 100)
