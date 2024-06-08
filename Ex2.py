import itertools
from pyvis.network import Network

import community as community_louvain  # Louvain method
from networkx.algorithms import community as nx_community
from itertools import combinations
import os
import json
import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def edge_selector_optimizer(G):
    """
    This function will select the edge with the highest betweenness centrality.
    """
    betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    return max(betweenness, key=betweenness.get)

# def edge_selector_optimizer_closeness(G):
#     closeness = nx.closeness_centrality(G, distance='weight')
#     edges = G.edges(data=True)
#     edge_centrality = {}
#     for u, v, data in edges:
#         edge_centrality[(u, v)] = closeness[u] + closeness[v]
#     return max(edge_centrality, key=edge_centrality.get)

def edge_selector_optimizer_degree(G):
    edges = G.edges(data=True)
    edge_centrality = {}
    for u, v, data in edges:
        edge_centrality[(u, v)] = G.degree[u] + G.degree[v]
    return max(edge_centrality, key=edge_centrality.get)

# def edge_selector_optimizer_eigenvector(G):
#     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, weight='weight')
#     edges = G.edges(data=True)
#     edge_centrality = {}
#     for u, v, data in edges:
#         edge_centrality[(u, v)] = eigenvector_centrality[u] + eigenvector_centrality[v]
#     return max(edge_centrality, key=edge_centrality.get)

# def edge_selector_optimizer_weight(G):
#     edges = G.edges(data=True)
#     edge_weights = {(u, v): data['weight'] for u, v, data in edges}
#     return max(edge_weights, key=edge_weights.get)

# def edge_selector_optimizer_pagerank(G):
#     pagerank = nx.pagerank(G, weight='weight')
#     edges = G.edges(data=True)
#     edge_centrality = {}
#     for u, v, data in edges:
#         edge_centrality[(u, v)] = pagerank[u] + pagerank[v]
#     return max(edge_centrality, key=edge_centrality.get)




def roll_over_cliques(G, start_clique, all_cliques, visited,k):
    """Roll over adjacent cliques starting from a specific 3-clique."""
    community = set(start_clique)
    stack = [start_clique]
    while stack:
        current_clique = stack.pop()
        visited.add(tuple(current_clique))
        # adjacent_cliques = find_adjacent_cliques(current_clique, all_cliques,k)
        adjacent_cliques = []
        clique_set = set(current_clique)
        for other in all_cliques:
            if len(clique_set.intersection(other)) == k - 1:
                adjacent_cliques.append(other)
        for adj_clique in adjacent_cliques:
            if tuple(adj_clique) not in visited:
                community.update(adj_clique)
                stack.append(adj_clique)
                visited.add(tuple(adj_clique))
    return community


def clique_percolation_k(G,k):
    """Perform the clique percolation algorithm for k on the graph."""
    cliques = list(nx.enumerate_all_cliques(G))
    all_cliques = [clique for clique in cliques if len(clique) == k]
    visited = set()
    communities = []

    for clique in all_cliques:
        if tuple(clique) not in visited:
            community = roll_over_cliques(G, clique, all_cliques, visited,k)
            communities.append(community)

    return communities


def calculate_modularity(G, communities):
    """Calculate the modularity of the graph based on the given communities manually."""
    m = G.size(weight='weight')  # Total number of edges in the graph (weighted)
    modularity = 0.0

    for community in communities:
        lc = 0  # Number of edges inside the community
        dc = 0  # Sum of the degrees of the nodes in the community

        for u in community:
            for v in community:
                if G.has_edge(u, v) and u != v:
                    lc += G[u][v].get('weight', 1.0)  # Count edges inside the community
            dc += G.degree(u, weight='weight')  # Sum of the degrees

        lc /= 2.0  # Each edge is counted twice
        modularity += (lc / m) - (dc / (2 * m)) ** 2

    return modularity
def community_detector(algorithm_name, network, most_valuable_edge=None):
    if algorithm_name not in ['girvin_newman', 'louvain', 'clique_percolation']:
        raise ValueError("Invalid algorithm name. Choose from 'girvin_newman', 'louvain', or 'clique_percolation'.")

    result = {}

    if algorithm_name == 'girvin_newman':
        # if most_valuable_edge is None:
        #     most_valuable_edge = edge_selector_optimizer

        best_partition = None
        best_modularity = -1

        for partition in nx_community.girvan_newman(network, most_valuable_edge):
            modularity = nx_community.modularity(network, partition)
            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = partition

        result['num_partitions'] = len(best_partition)
        result['partition'] = [list(c) for c in best_partition]
        result['modularity'] = best_modularity

    elif algorithm_name == 'louvain':
        # Louvain method inherently optimizes modularity
        best_partition = community_louvain.best_partition(network)
        communities = {}
        for node, comm in best_partition.items():
            communities.setdefault(comm, []).append(node)
        best_modularity = community_louvain.modularity(best_partition, network)

        result['num_partitions'] = len(communities)
        result['partition'] = list(communities.values())
        result['modularity'] = best_modularity

    elif algorithm_name == 'clique_percolation':
        best_partition = []
        best_modularity = -1
        for k in range(3, 10):
            communities = clique_percolation_k(G, k)
            modularity = calculate_modularity(G, communities)
            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = communities

        result['num_partitions'] = len(best_partition)
        result['partition'] = [list(c) for c in best_partition]
        result['modularity'] = best_modularity
    return result


def read_politician_names(file_path):
    """
    Read the central_political_players.csv file to map IDs to names.

    Parameters:
    - file_path: Path to the CSV file containing politician IDs and names.

    Returns:
    - id_to_name: A dictionary mapping user IDs to names.
    """
    id_to_name = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            id_to_name[row[0]] = row[1]
    return id_to_name
def construct_heb_edges(files_path, start_date, end_date, non_parliamentarians_nodes):
    # Read the CSV file containing central political players
    central_players_file = os.path.join('central_political_players_sna_ex2 - central_political_players.csv')
    central_players = set()
    with open(central_players_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            central_players.add(row[0])

    # Define the start and end dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Initialize the edge dictionary
    edge_dict = defaultdict(int)

    # Process each .txt file in the directory
    for filename in os.listdir(files_path):
        if filename.endswith('.txt'):
            # Extract the date from the filename
            file_date = datetime.strptime(filename.replace('.txt', '').replace('Hebrew_tweets.json.', ''), '%Y-%m-%d')
            if start_date <= file_date <= end_date:
                file_path = os.path.join(files_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        tweet = json.loads(line.strip())
                        if 'retweeted_status' in tweet:
                            user_u = tweet['user']['id_str']
                            user_v = tweet['retweeted_status']['user']['id_str']
                            if non_parliamentarians_nodes == 0:
                                if user_u in central_players and user_v in central_players:
                                    edge_dict[(user_u, user_v)] += 1
                                continue
                            edge_dict[(user_u, user_v)] += 1
    if non_parliamentarians_nodes==0:
        return edge_dict
    # Initialize the graph
    G = nx.DiGraph()

    # Add edges with weights to the graph
    for (user_u, user_v), weight in edge_dict.items():
        G.add_edge(user_u, user_v, weight=weight)

    # Calculate degree centrality for all nodes
    degree_centrality = nx.degree_centrality(G)

    # Select the top non_parliamentarians_nodes nodes based on degree centrality
    non_central_players = set(central_players)
    if non_parliamentarians_nodes > 0:
        sorted_nodes = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
        for node, centrality in sorted_nodes:
            if node not in central_players:
                non_central_players.add(node)
                if len(non_central_players) - len(central_players) >= non_parliamentarians_nodes:
                    break

    # Filter the graph to include only selected nodes
    filtered_G = G.subgraph(non_central_players).copy()

    # Construct the edge dictionary for the filtered graph
    filtered_edge_dict = defaultdict(int)
    for u, v, data in filtered_G.edges(data=True):
        filtered_edge_dict[(u, v)] = data['weight']

    return filtered_edge_dict


def construct_heb_network(edge_dict):
    """
    Construct a directed and weighted NetworkX graph from an edge dictionary.

    Parameters:
    - edge_dict: A dictionary where keys are tuples (user_u, user_v) representing directed edges
                 and values are the weights of these edges (e.g., retweet counts).

    Returns:
    - G: A NetworkX directed and weighted graph.
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges with weights to the graph
    for (user_u, user_v), weight in edge_dict.items():
        G.add_edge(user_u, user_v, weight=weight)

    return G


def visualize_graph(G, communities, id_to_name):
    """
    Visualize a directed and weighted NetworkX graph with nodes colored by community using PyVis.

    Parameters:
    - G: A NetworkX directed and weighted graph.
    - communities: A list of lists, where each list contains the nodes in a community.
    - id_to_name: A dictionary mapping user IDs to names.
    """
    net = Network(notebook=True, directed=True, cdn_resources='remote')

    # Generate a color list for the communities
    colormap = plt.colormaps.get_cmap('tab20')
    colors = colormap(range(50))

    # Create a dictionary to store node colors
    node_color_map = {}
    i = 0
    for community in communities:
        # color = next(itertools.cycle(colors))
        # print(color)
        i+=1
        color = colors[i]

        for node in community:
            node_color_map[node] = color

    # Add nodes and edges to the PyVis network
    for node in G.nodes:
        label = id_to_name.get(node, node)  # Use the name if available, otherwise the ID
        color = f'rgba({int(node_color_map[node][0] * 255)},{int(node_color_map[node][1] * 255)},{int(node_color_map[node][2] * 255)},1)'
        net.add_node(node, label=label, color=color, size=10)

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'])

    # Use ForceAtlas2 algorithm for layout
    net.force_atlas_2based()

    # Show the network
    net.show('graph_not_only_politicians.html')
# def visualize_graph(G, communities, separation_factor=10):
#     """
#     Visualize a directed and weighted NetworkX graph with nodes colored by community.
#
#     Parameters:
#     - G: A NetworkX directed and weighted graph.
#     - communities: A list of lists, where each list contains the nodes in a community.
#     - separation_factor: A factor to control the separation distance between communities.
#     """
#     # Generate a color map for the communities
#     colors = itertools.cycle(plt.cm.tab20.colors)  # You can choose any colormap here
#
#     # Create a dictionary to store node colors
#     node_color_map = {}
#     for community in communities:
#         color = next(colors)
#         for node in community:
#             node_color_map[node] = color
#
#     # Use a force-directed layout algorithm to position nodes
#     initial_pos = nx.spring_layout(G, k=0.15, iterations=20)
#
#     # Separate communities by adjusting positions
#     pos = {}
#     for i, community in enumerate(communities):
#         angle = 2 * 3.14159 * i / len(communities)
#         displacement = (separation_factor * i, separation_factor * i)
#         for node in community:
#             if node in initial_pos:
#                 pos[node] = [
#                     initial_pos[node][0] + displacement[0],
#                     initial_pos[node][1] + displacement[1]
#                 ]
#
#     # Draw nodes with community colors
#     for community in communities:
#         community_nodes = [node for node in community if node in G]
#         node_colors = [node_color_map[node] for node in community_nodes]
#         nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, node_size=100, node_color=node_colors)
#
#     # Draw edges with colors based on source node color
#     edge_colors = [node_color_map[u] for u, v in G.edges()]
#     nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, edge_color=edge_colors)
#
#     plt.title('Hebrew Political Tweets Network')
#     plt.show()

# Example usage
if __name__ == "__main__":
    G = nx.les_miserables_graph()

    # Run Girvan-Newman with the custom edge selector
    # result_gn = community_detector('girvin_newman', G, edge_selector_optimizer)
    # print("Girvan-Newman Result:", result_gn)
    #
    # # Run Louvain
    # result_louvain = community_detector('louvain', G)
    # print("Louvain Result:", result_louvain)

    #  Run Clique Percolation (without cdlib)
    # result_cp = community_detector('clique_percolation', G)
    # print("Clique Percolation Result:", result_cp)


    '''Example usage with different edge selector optimizers'''

    ### Closeness centrality
    # result_gn_closeness = community_detector('girvin_newman', G, edge_selector_optimizer_closeness)
    # print("Girvan-Newman with Closeness Centrality Result:", result_gn_closeness)
    #
    ### Degree Centrality
    # result_gn_degree = community_detector('girvin_newman', G, edge_selector_optimizer_degree)
    # print("Girvan-Newman with Degree Centrality Result:", result_gn_degree)
    #
    ### Eigenvector Centrality
    # result_gn_eigenvector = community_detector('girvin_newman', G, edge_selector_optimizer_eigenvector)
    # print("Girvan-Newman with Eigenvector Centrality Result:", result_gn_eigenvector)

    ### Weight Edge
    # result_gn_weight = community_detector('girvin_newman', G, edge_selector_optimizer_weight)
    # print("Girvan-Newman with Weight Result:", result_gn_weight)

    ### PageRank
    # result_gn_pagerank = community_detector('girvin_newman', G, edge_selector_optimizer_pagerank)
    # print("Girvan-Newman with PageRank Centrality Result:", result_gn_pagerank)
    '''2'''
    files_path = 'C:/Users/User/OneDrive/Data engineering/Year 3/Semester 2/Social networks/Ex 2/twitter_data_sna_ex2'  # Update with the correct path
    start_date = '2019-03-15'
    end_date = '2019-04-15'
    # non_parliamentarians_nodes = 0  # Example number of additional nodes to include
    non_parliamentarians_nodes = 20
    edge_dict = construct_heb_edges(files_path, start_date, end_date, non_parliamentarians_nodes)
    G = construct_heb_network(edge_dict)
    result_gn = community_detector('girvin_newman', G, edge_selector_optimizer_degree)
    print("Girvan-Newman Result:", result_gn)
    # print(G.edges(data=True))
    # Visualize the graph
    politician_names_file = 'central_political_players_sna_ex2 - central_political_players.csv'
    id_to_name = read_politician_names(politician_names_file)
    visualize_graph(G,result_gn['partition'],id_to_name)