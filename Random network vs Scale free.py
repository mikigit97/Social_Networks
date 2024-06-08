@staticmethod
def get_name():
    return "Mickael Zeitoun"

@staticmethod
def get_id():
    return "328783105"

import networkx as nx
import matplotlib.pyplot as plt

'''1.i'''
def random_networks_generator(n, p, num_networks=1, directed=False, seed=328783105):

    networks = []
    for _ in range(num_networks):
        if directed:
            G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
        else:
            G = nx.gnp_random_graph(n, p, seed=seed, directed=False)
        networks.append(G)
    return networks

def draw_networks(networks):
    """
    Draws each network in the list of networks.

    Parameters:
    - networks (list): A list of networkX graph objects.
    """
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    for i, network in enumerate(networks):
        plt.subplot(len(networks), 1, i + 1)  # Creates a subplot for each network
        nx.draw(network, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=16)
        plt.title(f'Network {i + 1}')
    plt.tight_layout()
    plt.show()

# networks = random_networks_generator(10, 0.3, num_networks=3, directed=True)

# This generates 3 directed graphs with 10 nodes each and an edge probability of 0.3, using a seed of 42.
# draw_networks(networks)
import networkx as nx
import numpy as np
'''1.ii'''
def network_stats(G):
    degrees = [d for n, d in G.degree()]
    degrees_avg = np.mean(degrees)
    degrees_std = np.std(degrees)
    degrees_min = np.min(degrees)
    degrees_max = np.max(degrees)

    if nx.is_connected(G):
        spl = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        spl = float('inf')  # Not connected, no finite shortest path
        diameter = float('inf')  # Not connected, infinite diameter

    stats = {
        'degrees_avg': degrees_avg,
        'degrees_std': degrees_std,
        'degrees_min': degrees_min,
        'degrees_max': degrees_max,
        'spl': spl,
        'diameter': diameter
    }

    return stats

# Example usage
# generated_networks = random_networks_generator(100,0.4,5)
# stats = network_stats(generated_networks[0])
# print(stats)

    #
'''1.iii'''
def networks_avg_stats(networks):
    # Initialize lists to hold values of each statistic
    avg_degrees = []
    std_degrees = []
    min_degrees = []
    max_degrees = []
    spls = []
    diameters = []

    # Gather statistics for each network
    for G in networks:
        stats = network_stats(G)
        avg_degrees.append(stats['degrees_avg'])
        std_degrees.append(stats['degrees_std'])
        min_degrees.append(stats['degrees_min'])
        max_degrees.append(stats['degrees_max'])
        spls.append(stats['spl'])
        diameters.append(stats['diameter'])

    # Calculate the average for each statistic
    avg_stats = {
        'degrees_avg': np.mean(avg_degrees),
        'degrees_std': np.mean(std_degrees),
        'degrees_min': np.mean(min_degrees),
        'degrees_max': np.mean(max_degrees),
        'spl': np.mean(spls),
        'diameter': np.mean(diameters)
    }

    return avg_stats


import networkx as nx
'''1.iv'''
def generate_network_types():
    types = {
        'a': {'num_networks': 20, 'n': 100, 'p': 0.1},
        'b': {'num_networks': 20, 'n': 100, 'p': 0.6},
        'c': {'num_networks': 10, 'n': 1000, 'p': 0.1},
        'd': {'num_networks': 10, 'n': 1000, 'p': 0.6}
    }

    results = {}
    for key, params in types.items():
        networks = random_networks_generator(params['n'], params['p'], num_networks=params['num_networks'], directed=False)
        avg_stats = networks_avg_stats(networks)
        results[key] = avg_stats
        print(f"Type {key}:")
        print(avg_stats)
        print("\n")

    return results

# Run the function
# network_types_stats = generate_network_types()
# print(network_types_stats)
import pickle
'''2.i'''
# Path to the pickle file
# filename = "C:/Users/User/OneDrive - post.bgu.ac.il/Data engineering/Year 3/Semester 2/Social networks/pickle network files/rand_nets.p"

# Load the list of networks from the pickle file
# with open(filename, 'rb') as file:
#     networks_list = pickle.load(file)

'''Verifying that I can access each network'''
# for i, network in enumerate(networks_list):
#     print(f"Network {i+1}:")
#     print(f"Number of nodes: {network.number_of_nodes()}")
#     print(f"Number of edges: {network.number_of_edges()}")
#     print("\n")
from scipy.stats import binom, ks_2samp
'''2.ii'''
def rand_net_hypothesis_testing(network, theoretical_p, alpha=0.05):
    n = network.number_of_nodes()
    observed_edges = network.number_of_edges()
    possible_edges = n * (n - 1) / 2
    k = observed_edges
    p = theoretical_p

    # Calculate the p-value manually for two-tailed test
    if k < p * possible_edges:
        p_value = 2 * binom.cdf(k, possible_edges, p)
    else:
        p_value = 2 * binom.sf(k-1, possible_edges, p)
    p_value = min(p_value, 1)  # Ensure p-value doesn't exceed 1 due to multiplication

    result = 'accept' if p_value > alpha else 'reject'
    return (p_value, result)

# Example usage:
# G = nx.gnp_random_graph(10, 0.1)
# test_result = rand_net_hypothesis_testing(G, 0.1)
# print("P-value:", test_result[0])
# print("Decision:", test_result[1])


'''2.iii'''
def most_probable_p(graph, p_values=[0.01, 0.1, 0.3, 0.6]):
    for p in p_values:
        rand = rand_net_hypothesis_testing(graph,p)
        if rand[1] == 'accept':
            result = p
            return result
    return -1

#     n = graph.number_of_nodes()
#     observed_degrees = [degree for node, degree in graph.degree()]
#
#     best_p = -1
#     min_ks_stat = float('inf')
#
#     for p in p_values:
#         expected_degrees = binom.rvs(n-1, p, size=10000)
#
#         ks_stat, p_val = ks_2samp(observed_degrees, expected_degrees)
#
#         if ks_stat < min_ks_stat:
#             min_ks_stat = ks_stat
#             best_p = p

    # return best_p if best_p != -1 else -1

# for i in networks_list:
#     print(most_probable_p(i, p_values=[0.01, 0.1, 0.3, 0.6]))


'''2.iv'''
def test_p_variation(network, base_p, variation=0.1):
    p_values = [0.01, 0.1, 0.3, 0.6]
    modified_ps = [(1 + variation) * base_p, (1 - variation) * base_p]
    results = {}

    for p in modified_ps:

        answer = rand_net_hypothesis_testing(network,p)

        results[f'p={p:.2f}'] = answer
    # print(f"Result given the optimal p:{rand_net_hypothesis_testing(network,base_p)}")
    return results


# Example usage with one of the networks
# chosen_network = networks_list[0]
# optimal_p = most_probable_p(chosen_network)
# results = test_p_variation(chosen_network, optimal_p)
# print("Results with adjusted p values:", results)

# larger_network = nx.gnp_random_graph(2000, optimal_p)
# larger_results = test_p_variation(larger_network, optimal_p)
# print("Results for a larger network:", larger_results)
#
# smaller_network = nx.gnp_random_graph(10, optimal_p)
# smaller_results = test_p_variation(smaller_network, optimal_p)
# print("Results for a smaller network:", smaller_results)

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


'''2.v'''
def plot_qq(network):
    # Calculate the degrees of the network
    degrees = np.array([deg for node, deg in network.degree()])

    # Standardize the degrees
    z = (degrees - np.mean(degrees)) / np.std(degrees)

    # Generate a QQ plot
    plt.figure(figsize=(6, 6))
    stats.probplot(z, dist="norm", plot=plt)
    plt.title('QQ Plot against Normal Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Degree Quantiles')
    plt.grid(True)
    plt.show()

# plot_qq(networks_list[0])


'''3.i'''
# Load the pickle file containing scale-free networks
# with open(r"C:/Users/User/OneDrive - post.bgu.ac.il/Data engineering/Year 3/Semester 2/Social networks/pickle network files/scalefree_nets.p", 'rb') as file:
#     scalefree_networks = pickle.load(file)
# #
# # Verify that the networks are loaded correctly
# for i, network in enumerate(scalefree_networks):
#     print(f"Network {i+1}:")
#     print(f"Number of nodes: {network.number_of_nodes()}")
#     print(f"Number of edges: {network.number_of_edges()}")
#     print("\n")

import powerlaw

# Example usage
# alphas = []
# for i, network in enumerate(scalefree_networks):
#     alpha = fit_powerlaw(network)
#     alphas.append(alpha)
#     print(f"Network {i + 1}: alpha = {alpha}")
#
# # Calculate the mean alpha if needed
# mean_alpha = sum(alphas) / len(alphas)
# print(f"Mean alpha: {mean_alpha}")


import numpy as np
import warnings

'''3.ii'''
def find_opt_gamma(network, treat_as_social_network=True):
    """
    Finds the optimal gamma parameter for a given network using the powerlaw package.

    Parameters:
    - network (networkX object): The network to analyze.
    - treat_as_social_network (bool): Whether to treat the network as a social network.

    Returns:
    - float: The optimal gamma parameter.
    """

    # Extract the degree sequence
    degrees = np.array([degree for node, degree in network.degree()])

    # Remove zero degrees if any
    degrees = degrees[degrees > 0]

    # Check for sufficient data points
    if len(degrees) < 30:
        warnings.warn("The network might be too small for a reliable power-law fit.")

    # Suppress specific warnings during fit
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        try:
            # Use discrete fitting if treating as a social network
            if treat_as_social_network:
                # Use discrete fitting if treating as a social network
                fit = powerlaw.Fit(degrees, discrete=True)
            else:
                # Use continuous fitting otherwise
                fit = powerlaw.Fit(degrees, discrete=False)

            # Return the alpha value, which corresponds to the gamma parameter
            gamma = fit.power_law.alpha
        except Exception as e:
            print(f"Error fitting power-law: {e}")
            return None

    return gamma


'''3.iii'''

# for i, network in enumerate(scalefree_networks):
#     gamma = find_opt_gamma(network)
#     print(f"Network {i + 1}: gamma = {gamma}")



'''3.iv'''
# scale_free_net = scalefree_networks[0]

# scale_free_stats = network_stats(scale_free_net)
# print("Scale-Free Network Stats:", scale_free_stats)
#
# n = scale_free_net.number_of_nodes()
# m = scale_free_net.number_of_edges()
# p = 2 * m / (n * (n - 1))  # Estimate p for similar number of edges
# er_net = nx.gnp_random_graph(n, p)
# er_net_stats = network_stats(er_net)
# print("Erdős-Rényi Network Stats:", er_net_stats)
#
# print("\nComparison of Network Statistics:")
# for key in scale_free_stats:
#     print(f"{key}: Scale-Free = {scale_free_stats[key]}, Erdős-Rényi = {er_net_stats[key]}")

'''4.i'''


# Load the pickle file containing mixed networks
# with open(r"C:/Users/User/OneDrive - post.bgu.ac.il/Data engineering/Year 3/Semester 2/Social networks/pickle network files/mixed_nets.p", 'rb') as file:
#     mixed_networks = pickle.load(file)

# Verify that the networks are loaded correctly
# for i, network in enumerate(mixed_networks):
#     print('here')
#     print(f"Network {i+1}:")
#     print(f"Number of nodes: {network.number_of_nodes()}")
#     print(f"Number of edges: {network.number_of_edges()}")
#     print("\n")
# Example usage to determine the type of each network in mixed_networks

'''4.ii'''
def netwrok_classifier(network):

    gamma = find_opt_gamma(network)

    if gamma is not None and 2 <gamma< 3:
        return 2  # Scale-Free Network
    else:
        return 1  # Random Network

'''4.iii'''
# for i, network in enumerate(mixed_networks):
#     stats = network_stats(network)
#     class_of_network = network_classifier(network)
#     gamma = find_opt_gamma(network)
#     if class_of_network == 2:
#         network_type = "Scale-Free"
#     else:
#         network_type = "Random"
#     print(f"Network {i + 1}:")
#     print(f"Statistics: {stats}")
#     print(f"Gamma: {gamma}")
#     print(f"Type: {network_type}")
#     print("\n")
