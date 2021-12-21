import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t


def findCeil(arr, r, l, h):
    while l < h:
        mid = (l + h) // 2
        if r > arr[mid]:
            l = mid + 1
        else:
            h = mid
    if arr[l] >= r:
        return l
    else:
        return -1


def myRand(freq, n):
    prefix = [0] * n
    prefix[0] = round(freq[0], 2)
    for i in range(n):
        prefix[i] = round(prefix[i - 1] + freq[i], 2)
    r = round(random.uniform(0, prefix[n - 1]), 2)
    indexc = findCeil(prefix, r, 0, n - 1)
    return indexc


def get_f_x(arr):
    res = []
    for i in range(len(arr) + 1):
        sm = 0
        for j in range(i):
            sm += arr[j]
        res.append(sm)
    return res


def get_square_m(values, probabilities):
    m = 0
    for c in range(len(values)):
        m += (values[c] ** 2) * probabilities[c]
    return m


def get_m_for_result(values):
    m = 0
    for c in range(len(values)):
        m += values[c]
    return m / len(values)


def get_m(values, probabilities):
    m = 0
    for c in range(len(values)):
        m += values[c] * probabilities[c]
    return m


def get_p_y(p):
    p_y = []
    for i in range(len(p)):
        column = 0
        for j in range(len(p)):
            column += p[j][i]
        p_y.append(column)
    return p_y


def get_d_for_result(values, m):
    n = len(values)
    d = 0
    for i in range(n):
        d += (values[i] - m) ** 2
    return d / (n - 1)


def get_interval_m(m, d, n, alpha):
    diff = np.sqrt(d) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    return m - diff, m + diff


def get_interval_d(d, n, alpha):
    left_part = n * d / chi2.isf((1 - alpha) / 2, n - 1)
    right_part = n * d / chi2.isf((1 + alpha) / 2, n - 1)
    return left_part, right_part


def get_r(x, mx, y, my, dx, dy):
    n = len(x)
    numerator = 0
    for i in range(n):
        numerator += (x[i] - mx) * (y[i] - my)
    return numerator / (np.sqrt(dx * dy * ((n - 1) ** 2)))


##-----------------------------------------------------------

P = [[0.1, 0.08, 0.01, 0.01, 0.03],
     [0.07, 0.06, 0.04, 0.03, 0.02],
     [0.04, 0.04, 0.05, 0.06, 0.05],
     [0.02, 0.05, 0.03, 0.02, 0.01],
     [0.03, 0.01, 0.07, 0.04, 0.03]]
X = [9, 7, 8, 5, 6]
Y = [2, 4, 1, 3, 5]

print("Теоретический закон распределения:\n", np.matrix(P))
P_X = [round(sum(P[j]), 2) for j in range(len(P))]
P_Y = get_p_y(P)
n = 10000
generatedX = []
generatedY = []
P_empiric = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
for i in range(n):
    x_k_index = myRand(P_X, len(X))
    x_k = X[x_k_index]
    generatedX.append(x_k)
    P_x_k = P_X[x_k_index]
    P_Y_x_k = [round(P[x_k_index][j] / P_x_k, 2) for j in range(len(P[x_k_index]))]
    y_x_k_index = myRand(P_Y_x_k, len(Y))
    y_x_k = Y[y_x_k_index]
    generatedY.append(y_x_k)
    P_empiric[x_k_index][y_x_k_index] = round(P_empiric[x_k_index][y_x_k_index] + 1 / n, 5)
print("Эмпирический закон распределения:\n", np.matrix(P_empiric))
m_theory_x = get_m(X, P_X)
print("-----------МатОжидание-------------")
P_X_empiric = [sum(P_empiric[j]) for j in range(len(P_empiric))]
P_Y_empiric = get_p_y(P_empiric)
print("Теоретическое M(X): ", m_theory_x)
m_result_x = get_m_for_result(generatedX)
print("M(X) для полученного закона распределения: ", m_result_x)
m_theory_y = get_m(Y, P_Y)
print("Теоретическое M(Y): ", m_theory_y)
m_result_y = get_m_for_result(generatedY)
print("M(Y) для полученного закона распределения: ", m_result_y)
d_theory_x = get_square_m(X, P_X) - (m_theory_x ** 2)
d_result_x = get_d_for_result(generatedX, m_result_x)
d_theory_y = get_square_m(Y, P_Y) - (m_theory_y ** 2)
d_result_y = get_d_for_result(generatedY, m_result_y)
print("-----------Дисперсия-------------")
print("Теоретическое D(X): ", d_theory_x)
print("D(X) для полученного закона распределения: ", d_result_x)
print("Теоретическое D(Y): ", d_theory_y)
print("D(Y) для полученного закона распределения: ", d_result_y)
print("-----------Интервальные оценки-------------")
interval_m_x = get_interval_m(m_result_x, d_result_x, len(generatedX), 0.05)
print("Интервальная оценка M(X):", interval_m_x)
interval_m_y = get_interval_m(m_result_y, d_result_y, len(generatedY), 0.05)
print("Интервальная оценка M(Y):", interval_m_y)
interval_d_x = get_interval_d(d_result_x, len(generatedX), 0.05)
print("Интервальная оценка D(X):", interval_d_x)
interval_d_y = get_interval_d(d_result_y, len(generatedY), 0.05)
print("Интервальная оценка D(Y):", interval_d_y)
r = get_r(generatedX, m_result_x, generatedY, m_result_y, d_result_x, d_result_y)
print("--------------------------------------------")

print("Коэффициент корреляции Пирсона:", r)



##-----------------------------------------------
x = X
y = P_X_empiric
fig, ax = plt.subplots()
ax.bar(x, y)
ax.set_xlabel("Вектор X")
ax.set_ylabel("Эмпирический P(X)")
plt.show()

x = Y
y = P_Y_empiric
fig, ax = plt.subplots()
ax.bar(x, y)
ax.set_xlabel("Вектор Y")
ax.set_ylabel("Эмпирический P(Y)")
plt.show()


