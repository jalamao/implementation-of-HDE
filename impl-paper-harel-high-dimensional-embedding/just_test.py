from __future__ import division, absolute_import, print_function
import numpy as np
from numpy.core.numeric import dot
from datetime import datetime
# from scipy.sparse import lil_matrix
# from scipy.sparse.csgraph import shortest_path
# from random import randint
# from sklearn.decomposition import PCA
# from algorithm import GraphDrawing
# from datetime import datetime

def k_center(distance_matrix, pivots, node_count):
    distances = []
    for i in range(1, node_count + 1):
        if i not in pivots:
            matrix_i = i - 1
            nearest_pivot = min(
                pivots,
                key=lambda p: distance_matrix[matrix_i, p - 1])
            nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
            distances.append((i, nearest_dis))

    return max(distances, key=lambda d: d[1])[0]



def average(a, axis=None, weights=None, returned=False):

    a = np.asanyarray(a)

    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size/avg.size)
    else:
        wgt = np.asanyarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


def covariance(x=None, bias = True):
    """        
    x: shape = (n_sample, n_feature) 
    bias: 
    return: X, X is cetered x and its dimensions are m*n
            S, S is covariance matrix, and its dimensions are m*m.
    """
    x = np.array(x)
    x = x.astype(float)
    n = x.shape[0]  # number of column of matrix, i.e., number of nodes.
    m = x.shape[1]  # number of column of matrix, i.e., number of pivots.
    print("number of nodes:", n)
    print("number of pivots:", m)
    r = average(x, axis=0)
    print("shape of mean vector:", x.shape)
    x -= r
    x = x.T  # transpose
    X = x    # X is centered x
    S = dot(x, x.T)
    if bias is False:
        S *= 1.0/float(n - 1)
    else:
        S *= 1.0 / float(n)
    print("dimensions of S:", S.shape)
    return X, S



def poweriteration(S=None, k=2, epsilon=0.001):
    """
    S: covariance matrix
    k: dimension of decomposition space
    epsilon: a constance
    U a list containing array of the first eigenvectors.
    """
    if S.shape[0] == S.shape[1]:
        m = S.shape[0]
    else:
        print("Invalid covariance matrix S.")
        return -1

    print("number of pivots:", m)
    U = []
    for i in range(k):
        i = i + 1
        ui_ = np.random.rand(m) # size of ui is m
        ui_ /= np.sqrt(dot(ui_, ui_)) # ui_=ui_/|ui_| normolized

        ui = ui_
        for j in range(i - 1):
            ui = ui - dot(ui, U[j])*U[j]
        ui_ = np.matmul(S, ui)
        ui_ /= np.sqrt(dot(ui_, ui_))  # ui=ui/|ui| normolization

        while dot(ui_, ui) < 1 - epsilon:
            ui = ui_

            for j in range(i - 1):
                ui = ui - dot(ui, U[j])*U[j]

            ui_ = np.matmul(S, ui)
            ui_ /= np.sqrt(dot(ui_, ui_))  # ui=ui/|ui| normolization

        U.append(ui_)  # store eigenvector into list U

    return U


def decomposition_space(X=None, U=None, k=2):
    """
    X: centered x, i.e., x subtracts mean of each dimension
    U: U stores eigenvector of x
    k: default value is 2.
    return: Y, i.e., new decomposition space and its dimensions are k.
    """
    Y = []
    for i in range(k):
         y = np.dot(X.T, U[i])
         Y.append(y)
    Y = np.array(Y).T
    print("shape of Y should be n*2:", Y.shape)
    return Y

def nomalization_distance_mtrix(distance_matrix=None):
    tmp = []
    distance_matrix = np.array(distance_matrix).T
    for i in distance_matrix:
        # i = np.array(i)
        max_i = max(i)
        min_i = min(i)
        i = (i - min_i) / (max_i - min_i)
        tmp.append(i)

    return np.array(tmp).T






##################### add follows by mty #########################

# class PivotMDS(object):
#     def __init__(self, distances=None, pivots=50, dim=2, **kwargs):
#         self.dst = np.array([m for m in distances])
#         self.n = len(self.dst)
#
#         if type(pivots) == type(1):  # type(1)="int"
#             self.k = pivots
#             self.pivots = np.random.permutation(len(self.dst))[:pivots]  # select randomly pivots. finally, self.pivots is list, e.g. pivots=[0, 3, 5, 8, 9, ...]
#             # self.pivots.sort()
#         elif type(pivots) == type([]):
#             self.pivots = pivots  # if pivots is a list, it denotes that pivots is specifed in list.
#             # self.pivots.sort()
#             self.k = len(self.pivots)
#         else:
#             raise AttributeError('pivots')
#
#     def optimize(self):
#         #        # Classical MDS (Torgerson)
#         #        J = identity(self.n) - (1/float(self.n))
#         #        B = -1/2. * dot(dot(J, self.dst**2), J)
#         #        w,v = linalg.eig(B)
#         #        tmp = zip([float(val) for val in w], range(self.n))
#         #        tmp.sort()
#         #        w1, w2 = tmp[-1][0], tmp[-2][0]
#         #        v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
#         #        return v1 * sqrt(w1), v2 * sqrt(w2)
#
#         # Pivot MDS
#         d = self.dst[[self.pivots]].T  # d:  shape=(n_sample, m_feature), m=number_of_pivots=k
#         C = d ** 2   # each elment of matrix d multiply itself.
#         # double-center d
#         cavg = np.sum(d, axis=0) / (self.k + 0.0)  # column sum of matrix d
#         ravg = np.sum(d, axis=1) / (self.n + 0.0)  # row sum of matrix d
#         tavg = np.sum(cavg) / (self.n + 0.0)  # total sum of matrix d
#         # TODO: optimize
#         for i in range(self.n):
#             for j in range(self.k):
#                 C[i, j] += -ravg[i] - cavg[j]  # each element of matrix C subtracts cavg and ravg.  C: shape=(n_sample, m_feature)
#
#         C = -0.5 * (C + tavg)  # C :  shape=(n_sample, m_feature)
#         w, v = np.linalg.eig(np.dot(C.T, C)) # w: array of eigenvalue, v: array of eigenvector, note that w[i] is corresponding to v[:, i].
#         tmp = zip([float(val) for val in w], range(self.n))
#         tmp.sort()
#         w1, w2 = tmp[-1][0], tmp[-2][0]
#         v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
#         x = np.dot(C, v1)
#         y = np.dot(C, v2)
#         return x, y


class PivotMDS(object):
    def __init__(self, d=None, pivots=50):   # d: shape=(n_sample, m_feature), m=number_of_pivots=k
        self.dst = np.array(d)
        self.n = len(self.dst)
        self.k = pivots
        # if type(pivots) == type(1):  # type(1)="int"
        #     self.k = pivots
        #     self.pivots = np.random.permutation(len(self.dst))[:pivots]  # select randomly pivots. finally, self.pivots is list, e.g. pivots=[0, 3, 5, 8, 9, ...]
        #     # self.pivots.sort()
        # elif type(pivots) == type([]):
        #     self.pivots = pivots  # if pivots is a list, it denotes that pivots is specifed in list.
        #     # self.pivots.sort()
        #     self.k = len(self.pivots)
        # else:
        #     raise AttributeError('pivots')

    def optimize(self):
        #        # Classical MDS (Torgerson)
        #        J = identity(self.n) - (1/float(self.n))
        #        B = -1/2. * dot(dot(J, self.dst**2), J)
        #        w,v = linalg.eig(B)
        #        tmp = zip([float(val) for val in w], range(self.n))
        #        tmp.sort()
        #        w1, w2 = tmp[-1][0], tmp[-2][0]
        #        v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
        #        return v1 * sqrt(w1), v2 * sqrt(w2)

        # Pivot MDS
        d = self.dst  # d:  shape=(n_sample, m_feature), m=number_of_pivots=k
        C = d ** 2   # each elment of matrix d multiply itself.
        # double-center d
        cavg = np.sum(d, axis=0) / (self.k + 0.0)  # column sum of matrix d
        ravg = np.sum(d, axis=1) / (self.n + 0.0)  # row sum of matrix d
        tavg = np.sum(cavg) / (self.n + 0.0)  # total sum of matrix d
        # TODO: optimize
        for i in range(self.n):
            for j in range(self.k):
                C[i, j] += -ravg[i] - cavg[j]  # each element of matrix C subtracts cavg and ravg.  C: shape=(n_sample, m_feature)

        C = -0.5 * (C + tavg)  # C :  shape=(n_sample, m_feature)
        w, v = np.linalg.eig(np.dot(C.T, C)) # w: array of eigenvalue, v: array of eigenvector, note that w[i] is corresponding to v[:, i].
        tmp = zip([float(val) for val in w], range(self.n))
        tmp.sort()
        w1, w2 = tmp[-1][0], tmp[-2][0]
        v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
        x = np.dot(C, v1)
        y = np.dot(C, v2)
        U = [x, y]
        U = np.array(U).T
        print("number of nodes:", self.n)
        print("shape of coordinary matrix of decomposition space:", U.shape)
        return U






if __name__ == "__main__":
    # pivots = np.random.permutation(10)[:5]
    # print(pivots)
    # pivots = np.sort(pivots)
    # print(list(pivots))
    #
    # def choose_pivots_randomly(dimension=50, number_nodes=None):
    #     pivots = np.random.permutation(number_nodes)[:dimension]
    #     pivots = np.sort(pivots)
    #     return pivots
    # l = [[1, 2, 3], [4, 5, 6]]
    # l = np.array(l)
    # print(l)
    # l[0][1] = 90
    # print(l)
    



    # import networkx as nx
    #
    # # G = nx.path_graph(4)
    # # nx.write_gml(G, 'test.gml')
    # H = nx.read_gml('data_gml/4elt.gml', label="id")
    # print(H)


    # m = [[1, 2, 9], [3, 4, 10], [8, 6, 9]]
    # m = np.array(m)
    # m = nomalization_distance_mtrix(m)
    # print(m)

    # dst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 7]]
    # dst = np.array(dst)
    # print(dst)
    # pivots = [0, 2]
    # d = dst[[pivots]].T
    # print(d)
    # print(d + 3)
    # w = [1, 2, 3]
    # n = 10
    # tmp = zip([float(val) for val in w], range(n))
    # print(tmp)
    # l = [3, 88, 0, 99, 55]
    # l = l.sort()
    # print(l)

    """
    result = np.genfromtxt("fake_data.csv",delimiter=',', dtype=[
            ('from', np.intp),
            ('to', np.intp),
            ('weight', np.float)
        ])


    node_count = max([
        *list(max(result, key=lambda i: i[0]))[0:2],
        *list(max(result, key=lambda i: i[1]))[0:2]])
    print("nodes:", node_count)
    # result = range(1, result+1)
    # result = np.array(result)
    graph_matrix = lil_matrix((node_count, node_count))
    for node_from, node_to, weight in result:
        graph_matrix[node_from - 1, node_to - 1] = weight
    # graph_matrix = np.array(graph_matrix)
    # print(graph_matrix[3, 3])
    distance_matrix = shortest_path(graph_matrix,
                             method='D',
                             directed=False,
                             unweighted=False)
    print("distance_matrix:")
    print(distance_matrix)
    # first_node = randint(0, 4)
    # # dimensions = 2
    # distances = []
    # print(first_node)
    # pivots = [first_node]  # firstly, select randomly the first pivot from V.
    # for i in range(0, 4):
    #     if i not in pivots:
    #         matrix_i = i - 1
    #         nearest_pivot = min(
    #             pivots,
    #             key=lambda p: distance_matrix[matrix_i, p - 1])
    #         nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
    #         distances.append((i, nearest_dis))
    # result = max(distances, key=lambda d: d[1])[0]
    # print(result)
    # pivots = [1]
    # nearest_pivot = min(
    #     pivots,
    #     key=lambda p: distance_matrix[1, p - 1])
    # print(nearest_pivot)
    pivot_nodes = [1, 2, 3]
    # result = k_center(distance_matrix, pivots, node_count)
    # print("k_center:", result)

    # distances = [(1, 2), (2, 5), (3, 90)]
    # m = max(distances, key=lambda d: d[1])[0]
    # print(m)
    # pivots = [1]
    # nearest_pivot = min(
    #     pivots,
    #     key=lambda p: distance_matrix[1, p - 1])



    node_range = range(1, node_count + 1)
    points = list(map(
        lambda i: tuple(distance_matrix[i - 1, p - 1]
                        for p in pivot_nodes),
        node_range
    ))
    # r = map(
    #     lambda i: tuple(distance_matrix[i - 1, p - 1]
    #                     for p in pivot_nodes),
    #     node_range
    # )
    print("points:")
    print(points)

    #

    #
    # points = [(1, 2, 3, 4), (5, 6, 7, 2), (1, 3, 4, 6)]
    # points = np.array(points)
    # print(points)
    pca = PCA(n_components=2)
    transformed_points = pca.fit_transform(points)
    print(transformed_points)
    """



    """
    graph_file = "maoty_new.csv"  # "3elt.csv"; "grid_50x50.csv"; "grid_100x100_corners_connected.csv"

    start_time = datetime.now()
    gd = GraphDrawing(dimension=50)  # default dimension is 50
    gd.transform(graph_file)
    gd.plot("figure_tests.png")  # drawing image is slow when solving for large number (1000's) of points
    end_time = datetime.now()
    time = (end_time - start_time).seconds
    print(time)
    """
    """
    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = dot(X, X_T.conj())
    c *= 1. / np.float64(fact)
    return c.squeeze()
    """

    #
    # x = [[10, 4, 4], [4, 6, 2]]
    #
    # x = np.array(x)
    # x = x.astype(float)
    # print(x)
    # r = average(x, axis=0)
    # print(r)
    #
    # x -= r # x=2*3
    # print(x)
    # result = dot(x, x.T)
    # print(result*(1.0/11))


    # u = [1.0, 2.0, 3.0]
    # u = np.array(u)
    # u /= np.sqrt(dot(u, u))
    # print(dot(u, u))
    # U = []
    # np.matmul()

    # a = [[1, 2], [4, 5]]
    # b = [[6, 7], [9, 2]]
    # a = np.array(a)
    # b = np.array(b)
    # # r = np.matmul(a, b)#dot(a, b)
    # # print(r.shape)
    # # print(r)
    # # for i in range(2):
    # U.append(a)
    # U.append(b)
    # print(U)
    # m = 3
    # l = [1, 2]
    # a = [[1, 2], [4, 5], [9, 8]]
    # l = np.array(l)
    # a = np.array(a)
    # r = dot(a, l)
    # l = np.resize(l, new_shape=(m, 1))
    # U = []
    # a = [(1, 2), (4, 5), (9, 8)]
    # # print(a.__len__())
    # # for i in range(a.__len__()):
    # #    U.append(np.array(a[i]))
    # #
    # # print(U)
    # # U = np.array(U)
    # # print(U)
    # a = np.array(a).T
    # print(a)
    # distances = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # dst = np.array([m for m in distances])
    # print(len(dst))
    # print(type(1))
    # pivots = 10
    # n = 20
    #
    # pivots = np.random.permutation(n)[:pivots]
    # print(pivots)


































