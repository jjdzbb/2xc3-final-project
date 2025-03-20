# All-pairs shortest path algorithms implementation

def floyd_warshall(graph):
    # Get all nodes
    nodes = graph.get_nodes()

    # Initialize distances and next matrices
    distances = {}
    next_vertex = {}

    for u in nodes:
        distances[u] = {}
        next_vertex[u] = {}
        for v in nodes:
            if u == v:
                distances[u][v] = 0
                next_vertex[u][v] = None
            else:
                weight = graph.get_edge_weight(u, v)
                if weight is not None:
                    distances[u][v] = weight
                    next_vertex[u][v] = v  # Next vertex after u is v
                else:
                    distances[u][v] = float('infinity')
                    next_vertex[u][v] = None

    # Floyd-Warshall algorithm
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if distances[i][k] != float('infinity') and distances[k][j] != float('infinity'):
                    if distances[i][j] > distances[i][k] + distances[k][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]
                        next_vertex[i][j] = next_vertex[i][k]

    # Compute previous (second-to-last vertex on the path)
    previous = {}
    for u in nodes:
        previous[u] = {}
        for v in nodes:
            if u == v or distances[u][v] == float('infinity'):
                previous[u][v] = None
            else:
                # Reconstruct the path
                path = []
                current = u
                while current != v and next_vertex[current][v] is not None:
                    path.append(current)
                    current = next_vertex[current][v]
                path.append(v)

                # The second-to-last vertex is the one before v in the path
                if len(path) > 2:
                    previous[u][v] = path[-2]
                else:
                    # If the path is just [u, v], the second-to-last vertex is u
                    previous[u][v] = u

    return distances, previous


def all_pairs_bellman_ford(graph):
    from shortest_path_algorithms import bellman_ford

    # Get all nodes
    nodes = graph.get_nodes()
    n = len(nodes)

    # Maximum valid k value for bellman_ford
    k = n - 2 if n > 2 else 1

    # Initialize distances and previous matrices
    distances = {}
    previous = {}

    # Run Bellman-Ford from each node
    for u in nodes:
        distances[u] = {}
        previous[u] = {}
        dist, paths = bellman_ford(graph, u, k)  # Use valid k value
        for v in nodes:
            distances[u][v] = dist[v]
            if u != v and dist[v] != float('infinity'):
                # If there's a path from u to v, the second-to-last vertex is the one before v in the path
                if len(paths[v]) > 1:
                    previous[u][v] = paths[v][-2]
                else:
                    # If the path is just [u, v], the second-to-last vertex is u
                    previous[u][v] = u
            else:
                previous[u][v] = None

    return distances, previous
