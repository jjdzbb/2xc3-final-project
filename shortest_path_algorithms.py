import heapq
import math


def dijkstra(graph, source, k):
    import heapq

    if k <= 0 or k >= graph.get_num_nodes() - 1:
        raise ValueError(
            f"k must be between 1 and {graph.get_num_nodes()-2} for this graph")

    # initialize distances with infinity for all nodes except source
    distances = {node: float('infinity') for node in graph.get_nodes()}
    distances[source] = 0

    # initialize paths
    paths = {node: [] for node in graph.get_nodes()}
    paths[source] = [source]

    # initialize relaxation counters
    relaxation_count = {node: 0 for node in graph.get_nodes()}

    # priority queue for Dijkstra's algorithm
    pq = [(0, source)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # skip if we've found a better path already
        if current_distance > distances[current_node]:
            continue

        # process all neighbors of the current node
        for neighbor, weight in graph.get_neighbors(current_node):
            # check if the node has been relaxed k times already
            if relaxation_count[neighbor] >= k:
                continue

            # calculate new distance
            distance = current_distance + weight

            # if we found a shorter path
            if distance < distances[neighbor]:
                # update distance
                distances[neighbor] = distance

                # update path
                paths[neighbor] = paths[current_node] + [neighbor]

                # increment relaxation counter
                relaxation_count[neighbor] += 1

                # add to priority queue
                heapq.heappush(pq, (distance, neighbor))

    return distances, paths


def bellman_ford(graph, source, k):
    if k <= 0 or k >= graph.get_num_nodes() - 1:
        raise ValueError(
            f"k must be between 1 and {graph.get_num_nodes()-2} for this graph")

    # initialize distances with infinity for all nodes except source
    distances = {node: float('infinity') for node in graph.get_nodes()}
    distances[source] = 0

    # initialize paths
    paths = {node: [] for node in graph.get_nodes()}
    paths[source] = [source]

    # initialize relaxation counters
    relaxation_count = {node: 0 for node in graph.get_nodes()}

    continue_iterations = True
    iteration_count = 0

    # maximum iterations would be k * number of nodes (worst case)
    max_iterations = k * graph.get_num_nodes()

    while continue_iterations and iteration_count < max_iterations:
        continue_iterations = False
        iteration_count += 1

        # check all edges
        for node in graph.get_nodes():
            for neighbor, weight in graph.get_neighbors(node):
                # check if the node has been relaxed k times already
                if relaxation_count[neighbor] >= k:
                    continue

                # calculate new distance
                if distances[node] != float('infinity'):
                    distance = distances[node] + weight

                    # if we found a shorter path
                    if distance < distances[neighbor]:
                        # update distance
                        distances[neighbor] = distance

                        # update path
                        paths[neighbor] = paths[node] + [neighbor]

                        # increment relaxation counter
                        relaxation_count[neighbor] += 1

                        # mark that we've made a change
                        continue_iterations = True

    return distances, paths
