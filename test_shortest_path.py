import unittest
import heapq
from graph_utils import Graph
from shortest_path_algorithms import dijkstra, bellman_ford


class TestShortestPathAlgorithms(unittest.TestCase):

    def test_simple_graph_dijkstra(self):

        graph = Graph()

        graph.add_edge(0, 1, 4)
        graph.add_edge(0, 2, 1)
        graph.add_edge(2, 1, 2)
        graph.add_edge(2, 3, 5)
        graph.add_edge(1, 3, 1)

        distances, paths = dijkstra(graph, 0, 1)


        expected_distances = {0: 0, 1: 4, 2: 1, 3: 6}
        expected_paths = {
            0: [0],
            1: [0, 1],
            2: [0, 2],
            3: [0, 2, 3] 
        }

   
        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

 
        distances, paths = dijkstra(graph, 0, 2)

        expected_distances = {0: 0, 1: 3, 2: 1, 3: 4}
        expected_paths = {
            0: [0],
            1: [0, 2, 1], 
            2: [0, 2],
            3: [0, 2, 1, 3] 
        }

      
        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

    def test_simple_graph_bellman_ford(self):
        graph = Graph()

       
        graph.add_edge(0, 1, 4)
        graph.add_edge(0, 2, 1)
        graph.add_edge(2, 1, 2)
        graph.add_edge(2, 3, 5)
        graph.add_edge(1, 3, 1)

     
        distances, paths = bellman_ford(graph, 0, 1)

        expected_distances_subset = {0: 0, 1: 4, 2: 1}
        expected_paths_subset = {
            0: [0],
            1: [0, 1],
            2: [0, 2]
        }

        self.assertEqual(distances[0], expected_distances_subset[0])
        self.assertEqual(distances[1], expected_distances_subset[1])
        self.assertEqual(distances[2], expected_distances_subset[2])

        self.assertEqual(paths[0], expected_paths_subset[0])
        self.assertEqual(paths[1], expected_paths_subset[1])
        self.assertEqual(paths[2], expected_paths_subset[2])


        self.assertTrue(distances[3] >= 4)

  
        distances, paths = bellman_ford(graph, 0, 2)

        expected_distances = {0: 0, 1: 3, 2: 1, 3: 4}
        expected_paths = {
            0: [0],
            1: [0, 2, 1],  
            2: [0, 2],
            3: [0, 2, 1, 3]  
        }

        self.assertEqual(distances[0], expected_distances[0])
        self.assertEqual(distances[1], expected_distances[1])
        self.assertEqual(distances[2], expected_distances[2])
        self.assertEqual(distances[3], expected_distances[3])

        self.assertEqual(paths[0], expected_paths[0])
        self.assertEqual(paths[1], expected_paths[1])
        self.assertEqual(paths[2], expected_paths[2])
        self.assertEqual(paths[3], expected_paths[3])

    def test_negative_weights_bellman_ford(self):
       
        graph = Graph()

        graph.add_edge(0, 1, 4)
        graph.add_edge(0, 2, 1)
        graph.add_edge(2, 1, -3) 
        graph.add_edge(2, 3, 5)
        graph.add_edge(1, 3, 1)

        
        distances, paths = bellman_ford(graph, 0, 2)

       
        expected_distances = {0: 0, 1: -2, 2: 1, 3: -1}
        expected_paths = {
            0: [0],
            1: [0, 2, 1], 
            2: [0, 2],
            3: [0, 2, 1, 3]  
        }

    
        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

    def test_complex_graph_multiple_paths(self):
  
        graph = Graph()



        graph.add_edge(0, 1, 3)
        graph.add_edge(0, 2, 4)
        graph.add_edge(0, 5, 2)
        graph.add_edge(1, 2, 2)
        graph.add_edge(2, 3, 5)
        graph.add_edge(2, 4, 3)
        graph.add_edge(3, 6, 4)
        graph.add_edge(4, 6, 2)
        graph.add_edge(5, 3, 1)


        dijkstra_distances, dijkstra_paths = dijkstra(graph, 0, 2)


        self.assertIn(6, dijkstra_distances)


        bellman_ford_distances, bellman_ford_paths = bellman_ford(graph, 0, 2)

        self.assertEqual(dijkstra_distances[6], bellman_ford_distances[6])


        dijkstra_distances_k3, dijkstra_paths_k3 = dijkstra(graph, 0, 3)

        expected_path_to_6 = [0, 5, 3, 6]
        expected_distance_to_6 = 7  

        self.assertEqual(dijkstra_distances_k3[6], expected_distance_to_6)
        self.assertEqual(dijkstra_paths_k3[6], expected_path_to_6)

    def test_disconnected_graph(self):

        graph = Graph()

        graph.add_edge(0, 1, 5)
        graph.add_edge(1, 2, 3)
        graph.add_edge(3, 4, 2)


        graph.add_node(5)
        graph.add_node(6)


        distances, paths = dijkstra(graph, 0, 2)


        expected_distances = {
            0: 0,
            1: 5,
            2: 8,
            3: float('infinity'),
            4: float('infinity'),
            5: float('infinity'),
            6: float('infinity')
        }


        self.assertEqual(distances[0], expected_distances[0])
        self.assertEqual(distances[1], expected_distances[1])
        self.assertEqual(distances[2], expected_distances[2])


        self.assertEqual(distances[3], float('infinity'))
        self.assertEqual(distances[4], float('infinity'))
        self.assertEqual(distances[5], float('infinity'))
        self.assertEqual(distances[6], float('infinity'))


        self.assertEqual(paths[3], [])
        self.assertEqual(paths[4], [])
        self.assertEqual(paths[5], [])
        self.assertEqual(paths[6], [])


        distances_bf, paths_bf = bellman_ford(graph, 0, 2)


        self.assertEqual(distances_bf[3], float('infinity'))
        self.assertEqual(paths_bf[3], [])

    def test_k_equal_to_one(self):

        graph = Graph()


        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)

        distances, paths = dijkstra(graph, 0, 1)


        self.assertEqual(distances[0], 0)
        self.assertEqual(distances[1], 1)

        if distances[2] != float('infinity'):
            self.assertTrue(distances[1] < distances[2])
        if distances[3] != float('infinity'):
            self.assertTrue(distances[2] < distances[3]
                            or distances[2] == float('infinity'))
        if distances[4] != float('infinity'):
            self.assertTrue(distances[3] < distances[4]
                            or distances[3] == float('infinity'))

        self.assertEqual(paths[0], [0])
        self.assertEqual(paths[1], [0, 1])

    def test_k_equal_to_n_minus_2(self):
        graph = Graph()

        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)

        distances, paths = dijkstra(graph, 0, 3)


        expected_distances = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        expected_paths = {
            0: [0],
            1: [0, 1],
            2: [0, 1, 2],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 3, 4]
        }

        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

    def test_negative_cycle_detection(self):
        graph = Graph()

        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 2)
        graph.add_edge(2, 3, 3)
        graph.add_edge(2, 1, -5)  

        distances, paths = bellman_ford(graph, 0, 2)

        self.assertTrue(distances[1] < 1)

    def test_standard_dijkstra_emulation(self):
        graph = Graph()

        for i in range(9):
            graph.add_edge(i, i+1, i+1)

        graph.add_edge(0, 5, 10)
        graph.add_edge(2, 7, 8)
        graph.add_edge(3, 9, 15)


        def standard_dijkstra(graph, source):
            distances = {node: float('infinity') for node in graph.get_nodes()}
            distances[source] = 0

            paths = {node: [] for node in graph.get_nodes()}
            paths[source] = [source]

            pq = [(0, source)]

            while pq:
                current_distance, current_node = heapq.heappop(pq)

                if current_distance > distances[current_node]:
                    continue

                for neighbor, weight in graph.get_neighbors(current_node):
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        paths[neighbor] = paths[current_node] + [neighbor]
                        heapq.heappush(pq, (distance, neighbor))

            return distances, paths

        std_distances, std_paths = standard_dijkstra(graph, 0)
        mod_distances, mod_paths = dijkstra(graph, 0, 8)
        self.assertEqual(std_distances, mod_distances)
        self.assertEqual(std_paths, mod_paths)

    def test_limited_relaxation_effect(self):
        graph = Graph()

        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(0, 4, 5)  

        results = {}
        for k in range(1, 4): 
            distances, paths = dijkstra(graph, 0, k)
            results[k] = (distances[4], paths[4])

        self.assertEqual(results[1][0], 5)
        self.assertEqual(results[1][1], [0, 4])
        self.assertEqual(results[3][0], 4)
        self.assertEqual(results[3][1], [0, 1, 2, 3, 4])
        self.assertTrue(results[1][0] >= results[2][0] >= results[3][0])


if __name__ == '__main__':
    unittest.main()
