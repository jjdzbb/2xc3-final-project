import unittest
from graph_utils import Graph
from all_pairs_shortest_path import floyd_warshall, all_pairs_bellman_ford


class TestAllPairsShortestPath(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()

        self.graph.add_edge(0, 1, 4)
        self.graph.add_edge(0, 2, 1)
        self.graph.add_edge(2, 1, 2)
        self.graph.add_edge(2, 3, 5)
        self.graph.add_edge(1, 3, 1)

        self.expected_distances = {
            0: {0: 0, 1: 3, 2: 1, 3: 4},
            1: {0: float('infinity'), 1: 0, 2: float('infinity'), 3: 1},
            2: {0: float('infinity'), 1: 2, 2: 0, 3: 3},
            3: {0: float('infinity'), 1: float('infinity'), 2: float('infinity'), 3: 0}
        }

        self.expected_previous = {
            0: {0: None, 1: 2, 2: 0, 3: 1},
            1: {0: None, 1: None, 2: None, 3: 1},
            2: {0: None, 1: 2, 2: None, 3: 1},
            3: {0: None, 1: None, 2: None, 3: None}
        }

    def test_floyd_warshall(self):
        distances, previous = floyd_warshall(self.graph)

        for u in self.graph.get_nodes():
            for v in self.graph.get_nodes():
                self.assertEqual(
                    distances[u][v], self.expected_distances[u][v])

        for u in self.graph.get_nodes():
            for v in self.graph.get_nodes():
                self.assertEqual(previous[u][v], self.expected_previous[u][v])

    def test_all_pairs_bellman_ford(self):
        distances, previous = all_pairs_bellman_ford(self.graph)

        for u in self.graph.get_nodes():
            for v in self.graph.get_nodes():
                self.assertEqual(
                    distances[u][v], self.expected_distances[u][v])

        for u in self.graph.get_nodes():
            for v in self.graph.get_nodes():
                self.assertEqual(previous[u][v], self.expected_previous[u][v])

    def test_negative_weights(self):
        graph = Graph()

        graph.add_edge(0, 1, 4)
        graph.add_edge(0, 2, 1)
        graph.add_edge(2, 1, -2)
        graph.add_edge(2, 3, 5)
        graph.add_edge(1, 3, 1)

        expected_distances = {
            0: {0: 0, 1: -1, 2: 1, 3: 0},
            1: {0: float('infinity'), 1: 0, 2: float('infinity'), 3: 1},
            2: {0: float('infinity'), 1: -2, 2: 0, 3: -1},
            3: {0: float('infinity'), 1: float('infinity'), 2: float('infinity'), 3: 0}
        }

        fw_distances, _ = floyd_warshall(graph)
        bf_distances, _ = all_pairs_bellman_ford(graph)

        for u in graph.get_nodes():
            for v in graph.get_nodes():
                self.assertEqual(fw_distances[u][v], expected_distances[u][v])

        for u in graph.get_nodes():
            for v in graph.get_nodes():
                self.assertEqual(bf_distances[u][v], expected_distances[u][v])


if __name__ == '__main__':
    unittest.main()
