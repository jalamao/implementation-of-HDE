import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class FR_Algorithm:

    def __init__(self, number_of_nodes=None, number_of_edges=None,
                 initial_temperature=100, cooling_factor=1,
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


    def apply_force_directed_algorithm(self, iteration=200, graph=None, coord_decomposition=None):
        self.initial_FR_position(coord_decomposition=coord_decomposition)
        # step = 0
        for i in range(iteration):
            # step += 1
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
    start_time = datetime.now()
    FR = FR_Algorithm(initial_temperature=100)
    graph = FR.get_graph_from_file("data/maotingyun.csv")
    nodes_num = FR.find_node_count(graph)
    # FR.number_of_nodes = nodes_num
    coors = FR.get_random_position(number_of_nodes=nodes_num)
    new_coords = FR.apply_force_directed_algorithm(iteration=100, graph=graph, coord_decomposition=coors)
    FR.plot(graph=graph, transformed_points=new_coords, filename="maotingyun.png")

    end_time = datetime.now()
    time = (end_time - start_time).seconds
    print(time)





        


