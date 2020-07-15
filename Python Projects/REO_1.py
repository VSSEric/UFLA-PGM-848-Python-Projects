print('=-'*78)
print(' '*60, 'REO 01 - LIST OF EXERCISE')
print('=-'*78)
print('COURSE: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS')
print('PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO')
print('STUDENT: ERIC VINICIUS VIEIRA SILVA')
print('DATE: 17/07/2020')
print('=-'*78)
print(' ')

print('-'*50)
print('Packages: numpy, math, matplotlib')
import numpy as np
import math
import matplotlib.pyplot as plt
print('-'*50)
print(' ')

print('EXERCISE 01:')
print(' ')
print('1.a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy:')
print('Answer:')
my_vector = np.array([43.5, 150.30, 17, 28, 35, 79, 20, 99.07, 15])
print('This is my vector:' + str(my_vector))
print('-'*50)
print(' ')

print('1.b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor')
print('Answer:')
dim_1 = len(my_vector)
mean_1 = np.mean(my_vector)
max_1 = max(my_vector)
min_1 = min(my_vector)
var_1 = np.var(my_vector)
print('The length of this vector is :' + str(dim_1))
print('The mean of this vector is :' + str(np.around(mean_1, 2)))
print('The max of this vector is :' + str(max_1))
print('The min of this vector is :' + str(min_1))
print('The variance of this vector is :' + str(np.around(var_1, 2)))
print('-'*50)
print(' ')

print("1.c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor "
      "declarado na letra a e o valor da média deste.")
print('Answer:')
my_new_vector = (my_vector - mean_1)**2
print("This is my new vector:" + str(np.around(my_new_vector, 2)))
print('-'*50)
print(' ')

print("1.d) Obtenha um novo vetor que contenha todos os valores superiores a 30")
print('Answer:')
bool_greater_30 = my_vector > 30
vector_30 = my_vector[bool_greater_30]
print("Vector > 30:" + str(vector_30))
print('-'*50)
print(' ')

print("1.e) Identifique quais as posições do vetor original possuem valores superiores a 30")
print('Answer:')
pos_greater_30 = np.where(my_vector > 30)
print("Vector positions where values are greater than 30:" + str(pos_greater_30))
print('-'*50)
print(' ')

print("1.f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.")
print('Answer:')
bool_1_5_last = [my_vector[0], my_vector[4], my_vector[-1]]
print("Vector 1, 5, and last positions:" + str(bool_1_5_last))
print('-'*50)
print(' ')

print("1.g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva "
      "posição durante as iterações")
print('Answer:')
it = 0
for i in range(0, len(my_vector), 1):
    it = it + 1
    print('Iteration: ' + str(it))
    print('The element ' + str(my_vector[i]) + ' is on position ' + str(i+1))

print('-'*50)
print(' ')

print("1.h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.")
print('Answer:')
it = 0
for i in range(len(my_vector)):
    it = it + 1
    print('Iteration: ' + str(it))
    print('The sum of squares of the first ' + str(i+1) + ' elements is equal to: '
          + str(np.around(sum(my_vector[:i+1]**2), 2)))

print('-'*50)
print(' ')

print("1.i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor")
print('Answer:')
pos = 0
while my_vector[pos] != 100:
    print("The element on position " + str(pos+1) + " is equal to " + str(my_vector[pos]))
    pos = pos+1
    if pos == (len(my_vector)):
        break

print('-'*50)
print(' ')

print("1.j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.")
print('Answer:')
seq = range(1, len(my_vector)+1, 1)
seq = np.array(seq)
print('The sequence is: ' + str(seq))
print('-'*50)
print(' ')

print("1.h) Concatene o vetor da letra a com o vetor da letra j.")
print('Answer:')
my_vector_seq = np.concatenate((my_vector, seq))
print('Vector and sequence concatenated: ' + str(my_vector_seq))
print('-'*50)
print(' ')

print('EXERCISE 02:')
print(' ')
print("2.a) Declare a matriz abaixo com a biblioteca numpy.")
print('Answer:')
my_matrix = np.array([[1, 3, 22], [2, 8, 18], [3, 4, 22], [4, 1, 23], [5, 2, 52], [6, 2, 18], [7, 2, 25]])
print('This is my matrix:')
print(my_matrix)
print('-'*50)
print(' ')

print("2.b) Obtenha o número de linhas e de colunas desta matriz")
print('Answer:')
nl,nc = np.shape(my_matrix)
print('Number of rows: ' + str(nl))
print('Number of columns: ' + str(nc))
print('-'*50)
print(' ')

print("2.c) Obtenha as médias das colunas 2 e 3.")
print('Answer:')
for i in range(1, nc, 1):
    print('The mean of column ' + str(i+1) + ' is equal to : ' + str(np.around(np.mean(my_matrix[:, i]), 2)))

print('-'*50)
print(' ')

print("2.d) Obtenha as médias das linhas considerando somente as colunas 2 e 3")
print('Answer:')
for i in range(0, nl, 1):
    print('Considering only columns 2 and 3, the mean of row ' + str(i+1) + ' is equal to: '
          + str(np.mean(my_matrix[[i], [1, 2]])))

print('-'*50)
print(' ')

print("2.e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota "
      "de severidade de uma doença e a terceira peso de 100 grãos.Obtenha os genótipos que possuem nota de severidade "
      "inferior a 5.")
print('Answer:')
score_less_5 = my_matrix[:, 1] < 5
my_matrix_less_5 = my_matrix[score_less_5]
print("The genotypes " + str(my_matrix_less_5[:, 0]) + " present score of disease severity less than 5")

print('-'*50)
print(' ')

print("2.f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de "
      "uma doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior"
      " ou igual a 22.")
print('Answer:')
weight_greater_equal_22 = my_matrix[:, 2] >= 22
my_matrix_greater_equal_22 = my_matrix[weight_greater_equal_22]
print("The genotypes " + str(my_matrix_greater_equal_22[:, 0]) + " present score of 100 grains greater or equal to 22")
print('-'*50)
print(' ')

print("2.g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma "
      "doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior"
      " a 3 e peso de 100 grãos igual ou superior a 22.")
print('Answer:')
score_to3_weight_22 = my_matrix[(my_matrix[:, 1] <= 3) & (my_matrix[:, 2] >= 22)]
print("The genotypes " + str(score_to3_weight_22[:, 0]) +
      " present score of disease severity less than or equal to 3 and score of 100 grains greater than or equal to 22:")
print('-'*50)
print(' ')

print("2.h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da "
      "matriz e o seu respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo "
      "repetido. Apresente a seguinte mensagem a cada iteração: Na linha X e na coluna Y ocorre o valor: Z."
      "Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25")

matrix_weight_25 = []
print('Answer:')
iteration = 0
for i in np.arange(0, nl, 1):
    for j in np.arange(0, nc, 1):
        iteration += 1
        print('Iteration: ' + str(iteration))
        print('In the row ' + str(i+1) +
              ' and column ' + str(j+1) +
              ' there is the value: ' + str(my_matrix[int(i), int(j)]))
        matrix_weight_25 = (my_matrix[:, 2] >= 25)
        matrix_25 = (my_matrix[matrix_weight_25])

print('-'*50)
print("The genotypes " + str(matrix_25[:, 0]) + " present score of 100 grains greater than or equal to 25")
print('-'*50)
print(' ')

print('EXERCISE 03:')
print(' ')
print("3.a) Crie uma função em um arquivo externo (outro arquivo .py) "
      "para calcular a média e a variância amostral um vetor qualquer, baseada em um lopp (for).")
print('Answer:')
from funcoesEric import summary
sampling = summary(my_vector, 4, 5)
print('-'*50)
print('Results:')
print('  Sample     Mean     Var')
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(str(sampling))
print('-'*50)
print(' ')

print("3.b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal"
      "com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação..")
print('Answer:')
array_10 = np.random.normal(100, math.sqrt(2500), 10)
array_100 = np.random.normal(100, math.sqrt(2500), 100)
array_1000 = np.random.normal(100, math.sqrt(2500), 1000)
print('array_10 = np.random.normal(100, math.sqrt(2500), 10)')
print('array_100 = np.random.normal(100, math.sqrt(2500), 100)')
print('array_1000 = np.random.normal(100, math.sqrt(2500), 1000)')
print('-'*50)
print(' ')

print("3.c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.")
print('Answer:')
print(' ')
print('Sampling of the array_10')
print('-'*50)
#from funcoesEric import summary
sampling2 = summary(array_10, 10, 1)
print('-'*50)
print('Results:')
print('  Sample     Mean     Var')
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(str(sampling2))
print('-'*50)
print(' ')

print('Sampling of the array_100')
print('-'*50)
#from funcoesEric import summary
sampling3 = summary(array_100, 100, 1)
print('-'*50)
print('Results:')
print('  Sample     Mean     Var')
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(str(sampling3))
print('-'*50)
print(' ')

print('Sampling of the array_1000')
print('-'*50)
#from funcoesEric import summary
sampling4 = summary(array_1000, 1000, 1)
print('-'*50)
print('Results:')
print('  Sample     Mean     Var')
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(str(sampling4))
print('-'*50)
print(' ')

print("3.d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.")
print('Answer:')
print('h10 = plt.hist(array_10)')
print('h100 = plt.hist(array_100)')
print('h1000 = plt.hist(array_1000)')
print('h10000 = plt.hist(np.random.normal(100, math.sqrt(2500), 10000))')

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(array_10, color="tab:red")
ax0.set_title('n = 10')
ax1.hist(array_100, color="tab:orange")
ax1.set_title('n = 100')
ax2.hist(array_1000, color="tab:green")
ax2.set_title('n = 1000')
ax3.hist(np.random.normal(100, math.sqrt(2500), 10000), color="tab:blue")
ax3.set_title('n = 10000')
fig.tight_layout()
plt.show()
print('-'*50)
print(' ')

print('EXERCISE 04:')
print(' ')
print("4.a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) "
      "quanto a quatro variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca "
      "numpy, apresente os dados e obtenha as informações de dimensão desta matriz.")
print('Answer:')
print(' ')
data_set = np.loadtxt('dados.txt')
print('DATA SET:')
print('    Gen    Rep    v1     v2      v3     v4    v5')
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(str(data_set))
print('-'*50)
print('Dimensions of the Data Set:')
nl, nc = data_set.shape
print('Number of rows: ' + str(nl))
print('Number of columns: ' + str(nc))
print('-'*50)
print(' ')

print("4.b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy")
print('Answer:')
print('-'*50)
print('help(np.unique):')
print('-'*50)
help(np.unique)
print('-'*50)
print(' ')
print('-'*50)
print('help(np.where):')
print('-'*50)
help(np.where)
print('-'*50)
print(' ')

print("4.c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas")
print('Answer:')
print("The evaluated genotypes were: " + str(np.unique(data_set[:, 0])))
print("The number of repeats was: " + str(int(max(data_set[:, 1]))))
print("Data of ( " + str(nc-2) + " ) variables were collected")
print('-'*50)
print(' ')

print("4.d) Apresente uma matriz contendo somente as colunas 1, 2 e 4")
print('Answer:')
data_set_v2 = data_set[:, [0, 1, 3]]
print('This is the Data set for variable 2 (column 4):')
print('   Gen    Rep     v2')
print(data_set_v2)
print('-'*50)
print(' ')

print("4.e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel "
      "da coluna 4. Salve esta matriz em bloco de notas.")
print('Answer:')
results = np.zeros((len(np.unique(data_set_v2[:, 0])), 5))
it = 0
for i in range(0, len(np.unique(data_set_v2[:, 0])), 1):
    it = it + 1
    print('-' * 50)
    print('Genotype: ' + str(it))
    print("Max: " + str(np.max((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2])))
    print("Min: " + str(np.min((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2])))
    print("Mean: " + str(np.around(np.mean((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2]), 2)))
    print("Var: " + str(np.around(np.var((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2]), 2)))
    results[i, 0] = i + 1
    results[i, 1] = np.max((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2])
    results[i, 2] = np.min((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2])
    results[i, 3] = np.around(np.mean((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2]), 2)
    results[i, 4] = np.around(np.var((data_set_v2[data_set_v2[:, 0] == i + 1])[:, 2]), 2)

np.savetxt('Result_Matrix.txt', results, fmt='%2.2f', delimiter='\t')
print(' ')
print('-' * 50)
print('Summary results matrix')
print('-' * 50)
print('     Gen       Max      Min     Mean     Var')
print(results)
print(' ')
print('-' * 50)

print("4.f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada "
      "na letra anterior.")
print('Answer:')
mean_greater_500 = results[:, 3] >= 500
results_500 = results[mean_greater_500]
print("The genotypes " + str(results_500[:, 0]) + " present means greater than or equal to 500")
print('-' * 50)
print(' ')

print("4.g) Apresente os seguintes graficos:")
print("Médias dos genótipos para cada variável. Utilizar o comando plt.subplot "
      "para mostrar mais de um grafico por figura:")

full_results = np.zeros((len(np.unique(data_set[:, 0])), 6))
it = 0
for i in range(0, len(np.unique(data_set[:, 0])), 1):
    it = it + 1
    full_results[i, 0] = i + 1
    full_results[i, 1] = np.around(np.mean((data_set[data_set[:, 0] == i + 1])[:, 2]), 2)
    full_results[i, 2] = np.around(np.mean((data_set[data_set[:, 0] == i + 1])[:, 3]), 2)
    full_results[i, 3] = np.around(np.mean((data_set[data_set[:, 0] == i + 1])[:, 4]), 2)
    full_results[i, 4] = np.around(np.mean((data_set[data_set[:, 0] == i + 1])[:, 5]), 2)
    full_results[i, 5] = np.around(np.mean((data_set[data_set[:, 0] == i + 1])[:, 6]), 2)

print(' ')
print('-' * 50)
print('Genotypes means for each of the variables')
print('-' * 50)
print('   Gen     v1     v2     v3     v4     v5')
print(full_results)
print(' ')
print('-' * 50)


plt.figure('Genotype means for each of the variables')
plt.subplot(2, 3, 1)
plt.bar(x=full_results[:, 0], height=full_results[:, 1], width=0.5, align='center', color="tab:red")
plt.title('Variable 1')
plt.xticks(full_results[:, 0])
plt.ylabel("mean")

plt.subplot(2, 3, 2)
plt.bar(x=full_results[:, 0], height=full_results[:, 2], width=0.5, align='center', color="tab:orange")
plt.title('Variable 2')
plt.xticks(full_results[:, 0])
plt.ylabel("mean")

plt.subplot(2, 3, 3)
plt.bar(x=full_results[:, 0], height=full_results[:, 3], width=0.5, align='center', color="tab:green")
plt.title('Variable 3')
plt.xticks(full_results[:, 0])
plt.ylabel("mean")

plt.subplot(2, 3, 4)
plt.bar(x=full_results[:, 0], height=full_results[:, 4], width=0.5, align='center', color="tab:blue")
plt.title('Variable 4')
plt.xticks(full_results[:, 0])
plt.ylabel("mean")

plt.subplot(2, 3, 5)
plt.bar(x=full_results[:, 0], height=full_results[:, 5], width=0.5, align='center', color="tab:purple")
plt.title('Variable 5')
plt.xticks(full_results[:, 0])
plt.ylabel("mean")
fig.tight_layout()
plt.show()

print("Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). "
      "No eixo X uma variável e no eixo Y outra.")

my_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
             'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

plt.style.use('ggplot')
plt.figure('2D Scatter Graph')

plt.subplot(2, 2, 1)
for i in np.arange(0, len(np.unique(data_set[:, 0])), 1):
    plt.scatter(full_results[i, 1], full_results[i, 2], s=50, alpha=0.8, label=int(full_results[i, 0]), c=my_colors[i])
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')

plt.subplot(2, 2, 2)
for i in np.arange(0, len(np.unique(data_set[:, 0])), 1):
    plt.scatter(full_results[i, 1], full_results[i, 3], s=50, alpha=0.8, label=int(full_results[i, 0]), c=my_colors[i])
plt.xlabel('Variable 1')
plt.ylabel('Variable 3')

plt.subplot(2, 2, 3)
for i in np.arange(0, len(np.unique(data_set[:, 0])), 1):
    plt.scatter(full_results[i, 2], full_results[i, 3], s=50, alpha=0.8, label=int(full_results[i, 0]), c=my_colors[i])
plt.xlabel('Variable 2')
plt.ylabel('Variable 3')

plt.legend(bbox_to_anchor=(2.08, 0.7), title='Genotypes', borderaxespad=0., ncol=5)
plt.show()

print('=-'*78)
print(' '*60, 'END OF THE REO 1 - LIST OF EXERCISE')
print('=-'*78)
