import time
import matplotlib.pyplot as plt
import numpy as np
from graph_utils import Graph, generate_random_graph
from shortest_path_algorithms import dijkstra, bellman_ford
import heapq
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def run_time_experiment(algorithm, graph, source, k):
    start_time = time.time()
    algorithm(graph, source, k)
    end_time = time.time()
    return end_time - start_time


def compare_k_values(algorithm, graph, source, k_values):
    results = {}
    for k in k_values:
        if k <= 0 or k >= graph.get_num_nodes() - 1:
            continue
        execution_time = run_time_experiment(algorithm, graph, source, k)
        results[k] = execution_time
    return results


def compare_graph_sizes(algorithm, sizes, density, source_ratio, k_ratio):
    results = {}
    for size in sizes:
        graph = generate_random_graph(size, density)
        source = int(size * source_ratio)
        k = max(1, int(size * k_ratio))
        if k >= size - 1:
            k = size - 2
        execution_time = run_time_experiment(algorithm, graph, source, k)
        results[size] = execution_time
    return results


def compare_graph_densities(algorithm, size, densities, source_ratio, k_ratio):
    results = {}
    for density in densities:
        graph = generate_random_graph(size, density)
        source = int(size * source_ratio)
        k = max(1, int(size * k_ratio))
        if k >= size - 1:
            k = size - 2
        execution_time = run_time_experiment(algorithm, graph, source, k)
        results[density] = execution_time
    return results


def compare_accuracy(graph, source, k_values):
    # first, get optimal distances with standard Dijkstra's algorithm (without k limit)
    def standard_dijkstra(graph, source):
        import heapq
        distances = {node: float('infinity') for node in graph.get_nodes()}
        distances[source] = 0

        # priority queue
        pq = [(0, source)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph.get_neighbors(current_node):
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    # get optimal distances
    optimal_distances = standard_dijkstra(graph, source)

    # results dictionaries
    dijkstra_results = {}
    bellman_ford_results = {}

    for k in k_values:
        if k <= 0 or k >= graph.get_num_nodes() - 1:
            continue

        # get distances with limited relaxations
        dijkstra_distances, _ = dijkstra(graph, source, k)
        bellman_ford_distances, _ = bellman_ford(graph, source, k)

        # calculate accuracy metrics (average distance error)
        # lower is better - 0 means perfect match with optimal
        dijkstra_accuracy = 0
        bellman_ford_accuracy = 0
        count = 0

        for node in graph.get_nodes():
            if node != source and optimal_distances[node] != float('infinity'):
                count += 1

                # for Dijkstra
                if dijkstra_distances[node] != float('infinity'):
                    # calculate error ratio (how much worse is our result vs optimal)
                    # 0 means perfect match, higher is worse
                    dijkstra_accuracy += (
                        dijkstra_distances[node] - optimal_distances[node]) / optimal_distances[node]
                else:
                    dijkstra_accuracy += 1.0  # penalize unreachable nodes

                # for Bellman-Ford
                if bellman_ford_distances[node] != float('infinity'):
                    bellman_ford_accuracy += (
                        bellman_ford_distances[node] - optimal_distances[node]) / optimal_distances[node]
                else:
                    bellman_ford_accuracy += 1.0  # penalize unreachable nodes

        if count > 0:
            dijkstra_results[k] = dijkstra_accuracy / count
            bellman_ford_results[k] = bellman_ford_accuracy / count
        else:
            dijkstra_results[k] = 0
            bellman_ford_results[k] = 0

    return dijkstra_results, bellman_ford_results


def run_all_experiments():
    results_dir = "experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    graph_sizes = [10, 20, 50, 100, 200]
    graph_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    k_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with suppress_output():
        dijkstra_size_results = compare_graph_sizes(
            dijkstra, graph_sizes, 0.3, 0, 0.5)
        bellman_ford_size_results = compare_graph_sizes(
            bellman_ford, graph_sizes, 0.3, 0, 0.5)

        dijkstra_density_results = compare_graph_densities(
            dijkstra, 100, graph_densities, 0, 0.5)
        bellman_ford_density_results = compare_graph_densities(
            bellman_ford, 100, graph_densities, 0, 0.5)

        graph = generate_random_graph(100, 0.3)
        source = 0
        k_values = [int(100 * ratio) for ratio in k_ratios]
        k_values = [k for k in k_values if 0 < k < 99]

        dijkstra_k_results = compare_k_values(
            dijkstra, graph, source, k_values)
        bellman_ford_k_results = compare_k_values(
            bellman_ford, graph, source, k_values)

        small_graph = generate_random_graph(20, 0.3)
        small_k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
        small_k_values = [k for k in small_k_values if 0 <
                          k < 19]  # filter valid k values

        dijkstra_accuracy, bellman_ford_accuracy = compare_accuracy(
            small_graph, 0, small_k_values)

    plt.figure(figsize=(10, 6))
    plt.plot(graph_sizes, [dijkstra_size_results[size]
             for size in graph_sizes], 'o-', label='Dijkstra')
    plt.plot(graph_sizes, [bellman_ford_size_results[size]
             for size in graph_sizes], 's-', label='Bellman-Ford')
    plt.xlabel('Graph Size (Number of Nodes)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Graph Size vs. Execution Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/graph_size_vs_time.png')

    plt.figure(figsize=(10, 6))
    plt.plot(graph_densities, [dijkstra_density_results[density]
             for density in graph_densities], 'o-', label='Dijkstra')
    plt.plot(graph_densities, [bellman_ford_density_results[density]
             for density in graph_densities], 's-', label='Bellman-Ford')
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Graph Density vs. Execution Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/graph_density_vs_time.png')

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [dijkstra_k_results[k]
             for k in k_values], 'o-', label='Dijkstra')
    plt.plot(k_values, [bellman_ford_k_results[k]
             for k in k_values], 's-', label='Bellman-Ford')
    plt.xlabel('K Value (Relaxation Limit)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('K Value vs. Execution Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/k_value_vs_time.png')

    plt.figure(figsize=(10, 6))
    k_for_plot = [
        k for k in small_k_values if k in dijkstra_accuracy and k in bellman_ford_accuracy]
    plt.plot(k_for_plot, [dijkstra_accuracy[k]
             for k in k_for_plot], 'o-', label='Dijkstra')
    plt.plot(k_for_plot, [bellman_ford_accuracy[k]
             for k in k_for_plot], 's-', label='Bellman-Ford')
    plt.xlabel('K Value (Relaxation Limit)')
    plt.ylabel('Error (Lower is Better)')
    plt.title('K Value vs. Algorithm Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/k_value_vs_accuracy.png')


if __name__ == "__main__":
    run_all_experiments()
