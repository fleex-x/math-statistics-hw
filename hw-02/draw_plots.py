import matplotlib.pyplot as plt
import numpy.random as rnd 
from typing import List

def gen_random_xs(rnd_gen, n: int) -> List[float]:
    return [rnd_gen() for i in range(n)]

def get_mk(xs: List[float], k: int) -> float:
    res = 0
    for x in xs:
        res += x ** k
    return res / len(xs)

def get_teta_for_uniform(xs: List[float], k: int) -> float:
    return (get_mk(xs, k) * (k + 1)) ** (1/k)

def get_teta_for_exponential(xs: List[float], k: int) -> float:
    factorial = 1
    for i in range(1, k):
        factorial *= i
    return (get_mk(xs, k) / factorial) ** (-1/k)    


def build_plot(rnd_gen, n: int, get_teta):

    def calc_mean_quadratic_error(k: int):
        error = 0
        for i in range(1000):
            xs = gen_random_xs(rnd_gen, n)
            error += (get_teta(xs, k) - 1) ** 2
        return error / 1000   

    ks = [i for i in range(1, n)]

    tetas = [calc_mean_quadratic_error(k) for k in ks]
    plt.plot(ks, tetas)
    plt.show()
    plt.close()


def uniform_distrib() -> float:
    return rnd.uniform(0, 1)

def exponentioal_distrib() -> float:
    return rnd.exponential(1)    



build_plot(exponentioal_distrib, 100, get_teta_for_exponential)
