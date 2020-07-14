########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# ALUNO: ERIC VINICIUS VIEIRA SILA
########################################################################################################################
# ARQUIVO EXTERNO CONTENDO MINHAS FUNÇÕES
import numpy as np
import random

import numpy as np


def summary(data, sample_size=None, rounds=None):
    data_vector = list(data)
    if sample_size is None:
        sample_size = len(data_vector)
    else:
        sample_size = sample_size

    #print('Data: ' + str(data_vector))
    print('Sample size: ' + str(sample_size))

    if rounds is None and sample_size <= 10:
        rounds = 3
    elif rounds is None and sample_size > 10:
        rounds = round(0.3*len(data_vector))
    else:
        rounds = rounds

    it = 0
    results = np.zeros((rounds, 3))
    for i in range(0, rounds, 1):
        print('-'*50)
        it = it+1
        print('Sampling: ' + str(it))
        #print('Data: ' + str(data_vector))
        print('Sample size: ' + str(sample_size))

        results[i, 0] = it

        sample = random.sample(data_vector, sample_size)
        #print('Sample: ' + str(sample))

        mean = np.mean(sample)
        results[i, 1] = mean

        variance = np.var(sample)
        results[i, 2] = variance

        print('Mean: ' + str(np.around(mean, 2)))
        print('Variance: ' + str(np.around(variance, 2)))

    return results
