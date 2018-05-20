from __future__ import division, absolute_import, print_function
from numpy.core.numeric import dot
import numpy as np
import matplotlib.pyplot as plt
import logging
# from FR import FR_Algorithm
from random import randint
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA, FastICA, KernelPCA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, SpectralEmbedding, MDS

from datetime import datetime


logging.basicConfig(filename='runtime.log',
                    format='%(levelname)s:%(message)s',
                    level=logging.INFO)

class PivotMDS(object):
    def __init__(self, d=None, pivots=50):
        # self.dst = np.array([m for m in d])
        self.dst = d   # d is a distance matrix and shape of d = (n_sample, n_sample).
        self.n = len(self.dst)

        if type(pivots) == type(1):  # type(1)="int"
            self.k = pivots
            self.pivots = np.random.permutation(len(self.dst))[:pivots]  # select randomly pivots. finally, self.pivots is list, e.g. pivots=[0, 3, 5, 8, 9, ...]
            # self.pivots.sort()
        elif type(pivots) == type([]):
            self.pivots = pivots  # if pivots is a list, it denotes that pivots is specifed in list.
            # self.pivots.sort()
            self.k = len(self.pivots)
        else:
            raise AttributeError('pivots')

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
        d = self.dst[[self.pivots]].T  # d:  shape=(n_sample, m_feature), m=number_of_pivots=k
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
        return U



#  Pivots MDS class
# class MDS(object):
#     def __init__(self, d=None, pivots=50):   # d: shape=(n_sample, m_feature), m=number_of_pivots=k
#         self.dst = np.array(d)
#         self.n = len(self.dst)
#         self.k = pivots
#         # if type(pivots) == type(1):  # type(1)="int"
#         #     self.k = pivots
#         #     self.pivots = np.random.permutation(len(self.dst))[:pivots]  # select randomly pivots. finally, self.pivots is list, e.g. pivots=[0, 3, 5, 8, 9, ...]
#         #     # self.pivots.sort()
#         # elif type(pivots) == type([]):
#         #     self.pivots = pivots  # if pivots is a list, it denotes that pivots is specifed in list.
#         #     # self.pivots.sort()
#         #     self.k = len(self.pivots)
#         # else:
#         #     raise AttributeError('pivots')
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
#         d = self.dst  # d:  shape=(n_sample, m_feature), m=number_of_pivots=k
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
#         U = [x, y]
#         U = np.array(U).T
#         print("number of nodes:", self.n)
#         print("shape of coordinary matrix of decomposition space:", U.shape)
#         return U



class GraphDrawing:
    def __init__(self, dimension=50, epsilon=0.01, version="PCA", normalization=False, kernel="linear", gamma=None, pivot_select =None,
                 fr_iteration=300, initial_temperature=60, cooling_factor=1, factor_attract=1, factor_repulsion=1, tsne_learning_rate=100, tsne_init="random"):
        """
        dimension:  number of pivots
        epsilon: condition of iteration termination while using version "PIT".
        version: "PCA", use the method of eigenvalue decomposition
                 "PIT", use the method of power iteration
        kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
            Kernel.Default = "linear".

        gamma : float, default=1/n_features
        Kernel coefficient for rbf and poly kernels. Ignored by other kernels.
        """
        self.dimension = dimension
        self.epsilon = epsilon
        self.version = version
        self.normalization = normalization
        self.gamma = gamma
        self.kpca_fun = kernel
        self.pivot_select = pivot_select   # if pivot_select="randomly", then select pivots randomly.
        # FR
        self.fr_iteration = fr_iteration
        self.initial_temperature = initial_temperature
        self.cooling_factor = cooling_factor
        self.factor_attract = factor_attract
        self.factor_repulsion = factor_repulsion

        #TSNE
        self.learning_rate = tsne_learning_rate
        self.init = tsne_init



    def transform(self, graph_file, first_node=None):
        logging.info('loading graph')
        """
        input: csv file of graph; formate: start_node, end_node, weight
        output: graph, a list, the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
        count amount of nodes from G
        """
        self.graph = self.load_graph(graph_file)  # obtain a array of graph


        self.node_count = self.find_node_count(self.graph)  # find the number of nodes in graph
        self.edge_count = len(self.graph)
        print("nodes:", self.node_count)
        print("edges:", self.edge_count)
        self.node_range = range(1, self.node_count + 1)

        logging.info('computing distance matrix')
        self.distance_matrix = self.compute_distance_matrix(self.graph, self.node_count)
        # self.distance_matrix = self.nomalization_distance_mtrix(distance_matrix=self.distance_matrix) # nomalized distance matrix
        ##############################  adjacency matrix ##########################################
        self.adjacency_matrix = self.get_adjacency_matrix(self.graph, self.node_count)
        ###########################################################################
        if first_node is None:
            """self.first_node = randint(0, self.node_count) + 1  # Choose the first pivot from V randomly."""
            self.first_node = randint(1, self.node_count)
        else:
            self.first_node = first_node  # Specify the first pivot.

        logging.info('finding pivots')
        """
        dimensions=m
        choose m pivots according to k-center.         
        """
        #####################################################
        if self.pivot_select == "randomly":
            self.pivot_nodes = self.choose_pivots_randomly(dimension=self.dimension, number_nodes=self.node_count)
        #####################################################
        else:
            self.pivot_nodes = self.choose_pivot_points(self.graph, self.dimension)  # self.pivot_nodes: a list

        logging.info('drawing graph in high dimensional space')
        """
        note that the number of pivot nodes is the same as dimension.
        formate of points:
        G=(V, E)
        |V|=n, dimensions = m = pivots
        d(vi, pj) denotes a distance computered by Dijkstra's algorithm in a G.

           p1          p2          p3     ...    pm
        v1 d(v1, p1) d(v1, p2)  d(v1, p3)     d(v1, pm)
        v2  .
        v3  .
        v4  .                                   .
        .   .                                   .
        .   .                                   .
        .   .
        vn d(vn, p1)       ...                d(vn, pm)


        """
        self.points = list(map(
            lambda i: tuple(self.distance_matrix[i - 1, p - 1]
                            for p in self.pivot_nodes),
            self.node_range
        ))

        if self.normalization is True:
            ##############################################################################################################
            self.points = self.nomalization_distance_mtrix(distance_matrix=self.points) # nomalized self.points
            ##############################################################################################################
        logging.info('project into a low dimension use PCA')

        if self.version == "HDE-SV":
            if self.dimension == 2:
                self.transformed_points = np.array(self.points)



        """
        PCA:
            input  array-like:  shape of self.points = (n_sample, n_feature)
            output array-like:  shape of self.transformed_points = (n_sample, n_component)

        """
        if self.version == "HDE":  # PCA denotes that algorithm uses PCA to decomposite original space.
            pca = PCA(n_components=2, copy=True)
            self.transformed_points = pca.fit_transform(self.points)

        if self.version == "HDE-Level":  # PCA denotes that algorithm uses PCA to decomposite original space.
            pca = PCA(n_components=3, copy=True)
            self.transformed_points = pca.fit_transform(self.points)
            pca = PCA(n_components=2, copy=True)
            self.transformed_points = pca.fit_transform(self.transformed_points)

        '''
          replace initial version as paper. by mty 2017-8-9
        '''
        if self.version == "HDE-PIT":  # PIT denotes that algorithm uses poweriteration to computer eigenvectors for decomposition space.
            X, S = self.covariance(self.points)
            # X = np.array(self.points).T
            # X = X.astype(float)
            U = self.poweriteration(S, epsilon=self.epsilon)
            self.transformed_points = self.decomposition_space(X, U)
            if self.node_count == (self.edge_count +1):   # determine wether it is a tree.
                FR = FR_Algorithm(number_of_nodes=self.node_count, initial_temperature=self.initial_temperature,
                                  cooling_factor=self.cooling_factor,
                                  factor_attract=self.factor_attract, factor_repulsion=self.factor_repulsion)
                # use FR to fine-tune
                self.transformed_points = FR.apply_force_directed_algorithm(iteration=self.fr_iteration, graph=self.graph,
                                                                             coord_decomposition=self.transformed_points)

        if self.version == "HDE-MDS":  # HDE-MDS denotes that algorithm combines with MDS.
            hde_mds = MDS()  # MDS object
            self.transformed_points = hde_mds.fit_transform(self.points)

        if self.version == "Pivot-MDS": # Pivot-MDS denotes that original version of Pivot MDS.
            pivot_mds = PivotMDS(d=self.distance_matrix, pivots=self.dimension)  # PivotMDS object
            self.transformed_points = pivot_mds.optimize()

        if self.version == "HDE-FICA":  # FICA denotes that algorithm uses Fast ICA to decomposite original space.
            #  fun, Could be either 'logcosh', 'exp', or 'cube'.
            fica = FastICA(n_components=2)
            # print(np.array(self.points).shape)
            self.transformed_points = fica.fit_transform(self.points)
            # print(np.array(self.transformed_points).shape)
            # FR = FR_Algorithm(number_of_nodes=self.node_count, initial_temperature=self.initial_temperature,
            #                   cooling_factor=self.cooling_factor, factor_attract=self.factor_attract, factor_repulsion=self.factor_repulsion)
            # # use FR to fine-tune
            # self.transformed_points = FR.apply_force_directed_algorithm(iteration=self.fr_iteration, graph=self.graph, coord_decomposition=self.transformed_points)

        if self.version == "HDE-KPCA": # FPCA denotes that algorithm uses kernel PCA to decomposite original space.
            kpca = KernelPCA(n_components=2, kernel=self.kpca_fun, gamma=self.gamma)
            self.transformed_points = kpca.fit_transform(self.points)

        if self.version == "HDE-NMF":
            nmf = NMF(n_components=2)
            self.transformed_points = nmf.fit_transform(self.points)

        if self.version == "HDE-TruncatedSVD":
            tsvd = TruncatedSVD(n_components=2)
            self.transformed_points = tsvd.fit_transform(self.points)

        if self.version == "HDE-LDA":
            lda = LinearDiscriminantAnalysis(n_components=2)
            y = []
            for i in range(self.node_count):
                y.append(1)
            y = np.array(y)
            lda = lda.fit(self.points, y=y)
            self.transformed_points = lda.transform(self.points)
        if self.version == "HDE-FR":

            pca = PCA(n_components=2, copy=True)
            self.transformed_points = pca.fit_transform(self.points)
            if self.node_count == (self.edge_count +1):   # determine wether it is a tree.
                FR = FR_Algorithm(number_of_nodes=self.node_count, initial_temperature=self.initial_temperature,
                                  cooling_factor=self.cooling_factor,
                                  factor_attract=self.factor_attract, factor_repulsion=self.factor_repulsion)
                # use FR to fine-tune
                self.transformed_points = FR.apply_force_directed_algorithm(iteration=self.fr_iteration, graph=self.graph,
                                                                             coord_decomposition=self.transformed_points)

        if self.version == "HDE-FICA-FR":

            fica = FastICA(n_components=2)
            self.transformed_points = fica.fit_transform(self.points)
            if self.node_count == (self.edge_count +1):   # determine wether it is a tree.
                FR = FR_Algorithm(number_of_nodes=self.node_count, initial_temperature=self.initial_temperature,
                                  cooling_factor=self.cooling_factor,
                                  factor_attract=self.factor_attract, factor_repulsion=self.factor_repulsion)
                # use FR to fine-tune
                self.transformed_points = FR.apply_force_directed_algorithm(iteration=self.fr_iteration, graph=self.graph,
                                                                             coord_decomposition=self.transformed_points)


        if self.version == "HDE-TSNE-FR":
            # pca = PCA(n_components=10, copy=True)
            # self.transformed_points = pca.fit_transform(self.points)
            tsne = TSNE(learning_rate= self.learning_rate, init=self.init)  # 'init' must be 'pca', 'random', or a numpy array
            self.transformed_points = tsne.fit_transform(self.points)
            if self.node_count == (self.edge_count + 1):  # determine wether it is a tree.
                FR = FR_Algorithm(number_of_nodes=self.node_count, initial_temperature=self.initial_temperature,
                                  cooling_factor=self.cooling_factor,
                                  factor_attract=self.factor_attract, factor_repulsion=self.factor_repulsion)
                # use FR to fine-tune
                self.transformed_points = FR.apply_force_directed_algorithm(iteration=self.fr_iteration,
                                                                            graph=self.graph,
                                                                            coord_decomposition=self.transformed_points)

        if self.version == "HDE-SPE":
            IP = SpectralEmbedding(n_components=2)
            self.transformed_points = IP.fit_transform(self.distance_matrix)
            # pca = PCA(n_components=2, copy=True)
            # self.transformed_points = pca.fit_transform( self.transformed_points)





        return self.node_count, self.edge_count



    # @staticmethod  # if you use key word "staticmethod", you can call the method like this: class_name.method,such as GraphDrawing.find_node_count.
    # def find_node_count(graph):  # find the number of nodes
    #     return max([
    #         *list(max(graph, key=lambda i: i[0]))[0:2],
    #         *list(max(graph, key=lambda i: i[1]))[0:2]])

    @staticmethod  # if you use key word "staticmethod", you can call the method like this: class_name.method,such as GraphDrawing.find_node_count.
    def find_node_count(graph=None):
        tmp = []
        for i in graph:
            if i[0] not in tmp:
                tmp.append(i[0])
            if i[1] not in tmp:
                tmp.append(i[1])
        return max(tmp)

    # original version altered by mty 2017-7-29
    @staticmethod
    def load_graph(graph_file):  # format of input file: from, to, weight. return: a array and the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
        return np.genfromtxt(graph_file, delimiter=',', dtype=[
            ('from', np.intp),
            ('to', np.intp),
            ('weight', np.float)
        ])

    # @staticmethod
    # def load_graph(graph_file):  # format of input file: from, to, weight. return: a array and the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
    #     return np.genfromtxt(graph_file, delimiter=',', dtype=[
    #         ('from', np.intp),
    #         ('to', np.intp),
    #         ('weight', np.float)
    #     ])

    def choose_pivot_points(self, graph, dimension):  # select pivots
        # pivots = []
        # coordinates = []
        pivots = [self.first_node]  # firstly, select randomly the first pivot from V.

        # find next by k-center problem
        for _ in range(0, dimension - 1):
            next_pivot = self.k_center(self.distance_matrix,
                                       pivots,
                                       self.node_count)
            pivots.append(next_pivot)

        return pivots  # a list containing pivots.

    @staticmethod
    def compute_distance_matrix(graph, node_count):
        graph_matrix = lil_matrix((node_count, node_count))

        for node_from, node_to, weight in graph:
            graph_matrix[node_from - 1, node_to - 1] = weight  # construct a distance matrix, graph_matrix
            """
            graph_matrix = nodes*nodes element=weight  so it is a sparse matrix.
               edge:
                 e.g.   v1, v3, 1
                        v1, v4, 1
                        ..........



                to:v1 v2 v3 ... vn
            from:
                 v1 0 weight(v1, v2)...weight(v1, vn)
                 v2
                 .
                 .
                 .
                 vn weiget(vn, v1)...weight(vn, vn-1) 0





            """

        # So, matrix of shortest path that we always mention consists of shortest path.
        return shortest_path(graph_matrix,
                             method='D',
                             directed=False,
                             unweighted=False)  # gain matrix of shortest path using Dijkstra's algorithm(Dijkstra' algorithm is one of the best algorithns of searching the shortest path)
##########################################################################################################
    @staticmethod
    def get_adjacency_matrix(graph, node_count):
        graph_matrix = lil_matrix((node_count, node_count))

        for node_from, node_to, weight in graph:
            graph_matrix[node_from - 1, node_to - 1] = weight  # construct a distance matrix, graph_matrix
            """
            graph_matrix = nodes*nodes element=weight  so it is a sparse matrix.
               edge:
                 e.g.   v1, v3, 1
                        v1, v4, 1
                        ..........



                to:v1 v2 v3 ... vn
            from:
                 v1 0 weight(v1, v2)...weight(v1, vn)
                 v2
                 .
                 .
                 .
                 vn weiget(vn, v1)...weight(vn, vn-1) 0





            """
        graph_matrix = graph_matrix.toarray()
        # So, matrix of shortest path that we always mention consists of shortest path.
        return graph_matrix
    ################################################################################################
    # select pivots from V
    # def k_center(self, distance_matrix, pivots, node_count=None):
    #     """
    #
    #
    #
    #     """
    #     distances = []
    #     for i in self.node_range:
    #         if i not in pivots:
    #             matrix_i = i - 1
    #             # suppose pivots have two elements, like [1, 2]. how to choose the third pivot? we can computer the distances between other nodes and pivots respectively. then record the min distance respectively. Finaly, chooose node that has max distance.
    #             nearest_pivot = min(
    #                 pivots,
    #                 key=lambda p: distance_matrix[
    #                     matrix_i, p - 1])  # return p that is notation coresponding to min value and distance_matrix[matrix_i, p - 1] is metric.
    #             nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
    #             distances.append((i, nearest_dis))
    #
    #     return max(distances, key=lambda d: d[1])[0]  # Compare every element from distances and select the max element.


    def k_center(self, distance_matrix, pivots, node_count=None):
        """



        """
        distances = []
        for i in self.node_range:
            if i not in pivots:
                matrix_i = i - 1
                # suppose pivots have two elements, like [1, 2]. how to choose the third pivot? we can computer the distances between other nodes and pivots respectively. then record the min distance respectively. Finaly, chooose node that has max distance.
                nearest_pivot = min(
                    pivots,
                    key=lambda p: distance_matrix[
                        matrix_i, p - 1])  # return p that is notation coresponding to min value and distance_matrix[matrix_i, p - 1] is metric.
                nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
                distances.append((i, nearest_dis))

        return max(distances, key=lambda d: d[1])[0]  # Compare every element from distances and select the max element.

    def plot(self, filename):
        # if self.normalization is True:
        #     ##############################################################################################################
        #     self.points = self.nomalization_distance_mtrix(distance_matrix=self.points) # nomalized self.points
        #     ##############################################################################################################
        fig = plt.figure(figsize=(6, 6))  # figsize=(w, h) in inches.
        ax = fig.add_subplot(111, frame_on=False)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # ax.xaxis.set_visible(True)
        # ax.yaxis.set_visible(True)

        logging.info('plot start')
        ax.scatter(self.transformed_points[:, 0],
                   self.transformed_points[:, 1],
                   s=1,
                   facecolor="black",
                   linewidth=0)

        for node_from, node_to, weight in self.graph:
            coord_from = list(self.transformed_points[node_from - 1])
            coord_to = list(self.transformed_points[node_to - 1])
            ax.plot([coord_from[0], coord_to[0]],
                    [coord_from[1], coord_to[1]],
                    linewidth=0.5,
                    color='black')

        logging.info('plot end')
        plt.savefig(filename, dpi=600)
        plt.close()

    ##################### add these by mty #######################################
    @staticmethod
    def average(a, axis=None, weights=None, returned=False):

        a = np.asanyarray(a)

        if weights is None:
            avg = a.mean(axis)
            scl = avg.dtype.type(a.size / avg.size)
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
                wgt = np.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
                wgt = wgt.swapaxes(-1, axis)

            scl = wgt.sum(axis=axis, dtype=result_dtype)
            if np.any(scl == 0.0):
                raise ZeroDivisionError(
                    "Weights sum to zero, can't be normalized")

            avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis) / scl

        if returned:
            if scl.shape != avg.shape:
                scl = np.broadcast_to(scl, avg.shape).copy()
            return avg, scl
        else:
            return avg


    def covariance(self, x=None, bias=True):
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
        r = self.average(x, axis=0)
        print("shape of mean vector:", x.shape)
        x -= r
        x = x.T  # transfer
        X = x  # X is centered x
        S = dot(x, x.T)
        if bias is False:
            S *= 1.0 / float(n - 1)
        else:
            S *= 1.0 / float(n)
        print("dimensions of S:", S.shape)
        return X, S

    ############### add it by mty #######################
    @staticmethod
    def poweriteration(S=None, k=2, epsilon=0.01):
        """
        S: covariance matrix
        k: dimension of decomposision space
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
            num = 0
            i = i + 1
            ui_ = np.random.rand(m) # size of ui is m
            # ui_ = np.ones(m)
            ui_ /= np.sqrt(dot(ui_, ui_)) # ui_=ui_/|ui_| normolized

            ui = ui_
            for j in range(i - 1):
                ui = ui - dot(ui, U[j])*U[j]
            ui_ = np.matmul(S, ui)
            ui_ /= np.sqrt(dot(ui_, ui_))  # ui=ui/|ui| normolization

            while dot(ui_, ui) < 1 - epsilon:
            # while iteration > 0:
            #     iteration = iteration - 1
                num = num + 1
                if num % 10 == 0:
                     print("loop")
                ui = ui_

                for j in range(i - 1):
                    ui = ui - dot(ui, U[j])*U[j]

                ui_ = np.matmul(S, ui)
                ui_ /= np.sqrt(dot(ui_, ui_))  # ui=ui/|ui| normolization


            U.append(ui_)  # store eigenvector into list U
            print("number of iteration:", num)

        return U

    ################# add it by mty  ####################
    @staticmethod
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
    ################# add it, by mty 2017-8-10   #############################
    @staticmethod
    def nomalization_distance_mtrix(distance_matrix=None):
        tmp = []
        distance_matrix = np.array(distance_matrix).T
        # distance_matrix = np.array(distance_matrix)
        for i in distance_matrix:
            # i = np.array(i)
            max_i = max(i)
            min_i = min(i)
            i = (i - min_i) / (max_i - min_i)
            tmp.append(i)

        return np.array(tmp).T
        # return np.array(tmp)

    # @staticmethod
    # def choose_pivots_randomly(dimension=50, number_nodes=None):
    #     pivots = np.random.permutation(number_nodes)[:dimension]
    #     # pivots = np.sort(pivots)
    #     return list(pivots)
    @staticmethod
    def choose_pivots_randomly(dimension=50, number_nodes=None):
        s = set()
        while s.__len__() < dimension:
            value = np.random.random_integers(1, number_nodes + 1)
            s.add(value)

        s = list(s)
        s = np.sort(s)
        return s



class FR_Algorithm:

    def __init__(self, number_of_nodes=None, number_of_edges=None,
                 initial_temperature=60, cooling_factor=1,
                 factor_attract=1, factor_repulsion=1,
                 MAX_X_DIMENSION=3600, MAX_Y_DIMENSION=3600):

        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        # self.number_of_iterations = number_of_iterations
        self.temperature = initial_temperature
        self.cooling_factor = cooling_factor
        self.factor_attract = factor_attract
        self.factor_repulsion =factor_repulsion
        self.MAX_X_DIMENSION = MAX_X_DIMENSION
        self.MAX_Y_DIMENSION = MAX_Y_DIMENSION
        self.pos = []


    def get_graph_from_file(self, graph_file=None):
        # graph, a list, the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
        graph = self.load_graph(graph_file)  # obtain a array of graph
        self.number_of_nodes = self.find_node_count(graph)
        return graph


    def get_random_position(self, number_of_nodes):
        # FR layout initial coordinates
        # find_node_count()
        coords = np.random.randint(0, 400, size=(number_of_nodes, 2))
        return coords

    def initial_FR_position(self, coord_decomposition=None):
        # pos = self.pos  # pos stores coordinates of decomposition space.
        for i in coord_decomposition:
            self.pos.append(i)

    def calculate_vertex_forces(self, initialgraph=None): # graph, array-like, shape=(n_sample, n_component), n_component=2 or 3
        """
        initialgraph: a list, the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)], each elenment is a edge, like "from to weight"
        coord_decomposition: array, shape=(n_sample, n_component)
        return: pos, an array, stores new coordinates in G.
        """
        MINIMUM_DISTANCE = 0.000001
        if self.factor_attract != 0:
            k_at = (1.0 / (self.factor_attract) * np.sqrt((self.MAX_X_DIMENSION * self.MAX_Y_DIMENSION) / float(self.number_of_nodes)))
            # print("number of nodes:", self.number_of_nodes)
        else:
            k_at = 0
        k_rep = self.factor_repulsion * np.sqrt((self.MAX_X_DIMENSION * self.MAX_Y_DIMENSION) / float(self.number_of_nodes))

        pos = self.pos  # pos stores coordinates of decomposition space.
        # disp = self.disp
        disp = []  # disp stores displacement of nodes in graph.
        for i in range(self.number_of_nodes):
            disp.append([0.0, 0.0])  # initial displacement.
        # k = np.sqrt((self.MAX_X_DIMENSION * self.MAX_Y_DIMENSION) / float(self.number_of_nodes))

        for i in range(self.number_of_nodes):

            # Calculate repulsion between all the elements in the same connected component
            if k_rep != 0:
                for h in range(self.number_of_nodes):
                    if h != i:
                        delta_x = pos[i][0] - pos[h][0]
                        delta_y = pos[i][1] - pos[h][1]
                        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
                        # if distance < (2 * k):
                        if distance == 0:
                            distance = MINIMUM_DISTANCE
                        disp[i][0] -= ((delta_x / distance) * (-(k_rep * k_rep) / distance))
                        disp[i][1] -= ((delta_y / distance) * (-(k_rep * k_rep) / distance))  # debug! if delta_y replace with delta_x, and it will always draw strainght line.


            # if k_rep != 0:
            #     for from_node in initialgraph:
            #         from_ = from_node[0]
            #         to_ = from_node[1]
            #         if from_ == (i + 1):  # labels of nodes begin with 1.
            #             delta_x = pos[i][0] - pos[to_ - 1][0]
            #             delta_y = pos[i][1] - pos[to_ - 1][1]
            #             distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
            #             if distance == 0:
            #                 distance = MINIMUM_DISTANCE
            #             disp[i][0] -= ((delta_x / distance) * (-(k_rep * k_rep) / distance))
            #             disp[i][1] -= ((delta_y / distance) * (-(k_rep * k_rep) / distance))  # debug! if delta_y replace with delta_x, and it will always draw strainght line.




            # Calculate attraction between connected elements
            if k_at != 0:
                for from_node in initialgraph:
                    from_ = from_node[0]
                    to_ = from_node[1]
                    if from_ == (i + 1):  # labels of nodes begin with 1.
                        delta_x = pos[i][0] - pos[to_ - 1][0]
                        delta_y = pos[i][1] - pos[to_ - 1][1]
                        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
                        if distance == 0:
                            distance = MINIMUM_DISTANCE
                            # print("kkkkk")

                        disp[i][0] -= (delta_x / distance) * ((distance * distance) / (k_at ))
                        disp[i][1] -= (delta_y / distance) * ((distance * distance) / (k_at ))
                        disp[to_ - 1][0] += (delta_x / distance) * ((distance * distance) / (k_at ))
                        disp[to_ - 1][1] += (delta_y / distance) * ((distance * distance) / (k_at ))




        # Update the positions according to the forces calculated before and the current temperature
        for i in range(self.number_of_nodes):
            disp_x = disp[i][0]
            disp_y = disp[i][1]
            module_disp = np.sqrt(disp_x * disp_x + disp_y * disp_y)
            if module_disp != 0:
                pos[i][0] += ((disp[i][0] / module_disp) * min(module_disp, self.temperature))
                pos[i][1] += ((disp[i][1] / module_disp) * min(module_disp, self.temperature))


        self.pos = pos

    # cooling function
    def cooling(self):
        if self.temperature > 1:
            self.temperature -= 0.2 * self.cooling_factor
        else:
            if self.temperature > 0:
                self.temperature -= 0.01
            else:
                self.temperature = 0

    def apply_force_directed_algorithm(self, iteration=100, graph=None, coord_decomposition=None):
        self.initial_FR_position(coord_decomposition=coord_decomposition)
        for i in range(iteration):
            self.cooling()
            self.calculate_vertex_forces(initialgraph=graph)
        print("FR Iteration finished!")

        return np.array(self.pos)

    @staticmethod
    def load_graph(graph_file):  # format of input file: from, to, weight. return: a array and the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
        return np.genfromtxt(graph_file, delimiter=',', dtype=[
            ('from', np.intp),
            ('to', np.intp),
            ('weight', np.float)
        ])

    @staticmethod  # if you use key word "staticmethod", you can call the method like this: class_name.method,such as GraphDrawing.find_node_count.
    def find_node_count(graph=None):
        tmp = []
        for i in graph:
            if i[0] not in tmp:
                tmp.append(i[0])
            if i[1] not in tmp:
                tmp.append(i[1])
        return max(tmp)

    def plot(self, graph=None, transformed_points=None, filename=None):
        fig = plt.figure(figsize=(6, 6))  # figsize=(w, h) in inches.
        ax = fig.add_subplot(111, frame_on=False)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


        ax.scatter(transformed_points[:, 0],
                   transformed_points[:, 1],
                   s=1,
                   facecolor="black",
                   linewidth=0)

        for node_from, node_to, weight in graph:
            coord_from = list(transformed_points[node_from - 1])
            coord_to = list(transformed_points[node_to - 1])
            ax.plot([coord_from[0], coord_to[0]],
                    [coord_from[1], coord_to[1]],
                    linewidth=0.5,
                    color='black')

        plt.savefig(filename, dpi=600)
        plt.close()




if __name__ == "__main__":

    graph_file = "neuralnet.csv"        # "3elt.csv"   ; "grid_50x50.csv"  ;  "grid_100x100_corners_connected.csv"  ;   "tests.csv"   ;   "tri_mesh_6"  ;  "4elt.csv"
    # "HDE"  ;  "HDE-PIT"   ;  "HDE-MDS"   ;  "Pivot-MDS"  ;   "HDE-FICA-FR"  ;  "HDE-KPCA"  ;  "HDE-NMF"  ;  "HDE-TruncatedSVD"  ;  "HDE-LDA"  ;  "HDE-FR" ; "HDE-FICA-FR"
    version = "HDE-FICA"
    start_time = datetime.now()
    normalization = False
    """
    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
    Kernel.Default = "linear".
    
    gamma : float, default=1/n_features while None.
    Kernel coefficient for rbf and poly kernels. Ignored by other
    kernels.
    """
    gd = GraphDrawing(dimension=5, epsilon=0.01, version=version, normalization=normalization, kernel= "sigmoid" , gamma=None, fr_iteration=400, initial_temperature=100, pivot_select="andomly", tsne_learning_rate=300, tsne_init="pca")  # default dimension is 50  , pivot_select="randomly"

    graph_name = "images/mtyfigure1_" + graph_file + ".png"
    nodes, edges = gd.transform('data/' + graph_file)
    end_computation = datetime.now()
    computation_time = (end_computation - start_time).seconds
    print("time of computation:", computation_time)
    gd.plot(graph_name)  # drawing image is slow when solving for large number (1000's) of points
    end_plot = datetime.now()
    plot_time = (end_plot - end_computation).seconds
    print("time of plot:", plot_time)

    end_time = datetime.now()
    time = (end_time - start_time).seconds

    with open("cost_time.txt", "a") as f:
        if normalization is True:
            normalization = "Ture"
        else:
            normalization = 'False'
        f.write("version:" + version + "  " + "normalization:" + normalization + "  " + graph_file + ": " +
                "nodes=" + str(nodes) + "  edges=" + str(edges) +  "  time_of_computation=" + str(computation_time) +
                "  time_of_plot=" + str(plot_time) + "  total_time=" + str(time) + "   name of graph:" + graph_name + "\n")
    print("total time:", time)


