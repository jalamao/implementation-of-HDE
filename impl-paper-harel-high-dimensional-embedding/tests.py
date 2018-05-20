import unittest
from algorithm import GraphDrawing
from random import randint


class TestGraphDrawing(unittest.TestCase):
    """
    Testcase for class GraphDrawing.
    The test data is a circle of 99 nodes. Any edge's weight is 1.
    """

    @classmethod
    def setUpClass(cls):
        cls.drawer = GraphDrawing(dimension=8)
        cls.drawer.transform('data/tests.csv', first_node=1)

    def test_graph(self):
        """
        Graph data should be read into a matrix row by row.
        """
        test_row = [randint(1, 98) for _ in range(0, 10)]
        for row in test_row:
            self.assertEqual(tuple(self.drawer.graph[row - 1]), (row, row + 1, 1.0))

    def test_max_node(self):
        """
        The max node should be 99
        """
        self.assertEqual(self.drawer.node_count, 99)

    def test_distance_matrix(self):
        """
        The distance from i-th node to (i+1)-th node should be 1.
        The distance from the last node (99) to the first node (1) should be 1.
        """
        distance_matrix = self.drawer.distance_matrix
        self.assertEqual(distance_matrix[1, 2], 1)
        self.assertEqual(distance_matrix[1, 3], 2)
        self.assertEqual(distance_matrix[98, 0], 1)
        self.assertEqual(distance_matrix[98, 1], 2)

    def test_pivot_nodes(self):
        """
        Select 8 pivot nodes from a 99 node circle.
        These pivot nodes should evenly distributed on the circle.
        """
        self.assertEqual(self.drawer.pivot_nodes,
                         [1, 50, 75, 25, 13, 37, 62, 87])

    def test_high_dimensional_points(self):
        """
        The i-th pivot's coordinate should has a 0.0 on i-th axie.
        """
        for index, pivot in enumerate(self.drawer.pivot_nodes):
            self.assertEqual(
                self.drawer.points[pivot - 1][index],
                0.0)


if __name__ == '__main__':
    # unittest.main()
    gd = GraphDrawing()  # default dimension is 50
    gd = gd.transform('data/snowflake_A.csv')

    # now you can retrieve tranformed points by accessing `gd.transformed_points`.

    gd.plot("figure.png")  # drawing image is slow when solving for large number (1000's) of points
