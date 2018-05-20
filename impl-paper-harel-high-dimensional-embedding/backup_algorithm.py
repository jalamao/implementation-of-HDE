from __future__ import division, absolute_import, print_function
from numpy.core.numeric import dot
import numpy as np
import matplotlib.pyplot as plt
import logging
# from math import inf
from random import randint
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA

logging.basicConfig(filename='runtime.log',
                    format='%(levelname)s:%(message)s',
                    level=logging.INFO)


class GraphDrawing:
    def __init__(self, dimension=50, epsilon=0.01, version="PCA"):
        """
        dimension:  number of pivots
        epsilon: condition of iteration termination
        version: "PCA", use the method of eigenvalue decomposition
                 "PIT", use the method of power iteration
        """
        self.dimension = dimension
        self.epsilon = epsilon
        self.version = version
    def transform(self, graph_file, first_node=None):
        logging.info('loading graph')
        """
        input: csv file of graph; formate: start_node, end_node, weight
        output: a array, the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
        count amount of nodes from G
        """
        self.graph = self.load_graph(graph_file)  # obtain a array of graph
        self.node_count = self.find_node_count(self.graph)  # find the number of nodes in graph
        self.node_range = range(1, self.node_count + 1)

        logging.info('computing distance matrix')
        self.distance_matrix = self.compute_distance_matrix(self.graph,
                                                            self.node_count)

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
        self.pivot_nodes = self.choose_pivot_points(self.graph, self.dimension)

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

        logging.info('project into a low dimension use PCA')
        """
        PCA:
            input  array-like:(n_sample, n_feature)
            output array-like:(n_sample, n_component)

        """
        if self.version == "PCA":
            pca = PCA(n_components=2, copy=True)
            self.transformed_points = pca.fit_transform(self.points)

        '''
          replace initial version as paper. by mty 2017-8-9
        '''
        if self.version == "PIT":
            X, S = self.covariance(self.points)
            # X = np.array(self.points).T
            # X = X.astype(float)
            U = self.poweriteration(S, epsilon=self.epsilon)
            self.transformed_points = self.decomposition_space(X, U)



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
    def load_graph(
            graph_file):  # format of input file: from, to, weight. return: a array and the elements are tuples, like [(1, 2, 1) (3, 1, 1) (2, 3, 1)]
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
                 v1 0 weiget(v1, v2)...weight(v1, vn)
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

    # select pivots from V
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
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, frame_on=False)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

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





if __name__ == "__main__":
    pass
