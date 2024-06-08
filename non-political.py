import os
import json
import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt


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


def visualize_graph(G):
    """
    Visualize a directed and weighted NetworkX graph without node and edge descriptions.

    Parameters:
    - G: A NetworkX directed and weighted graph.
    """
    pos = nx.spring_layout(G, k=0.15, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=2, edge_color='red')

    plt.title('Hebrew Political Tweets Network')
    plt.show()
# Example usage
files_path = 'C:/Users/User/OneDrive/Data engineering/Year 3/Semester 2/Social networks/Ex 2/twitter_data_sna_ex2'  # Update with the correct path
start_date = '2019-03-15'
end_date = '2019-04-15'
non_parliamentarians_nodes = 20 # Example number of additional nodes to include
edge_dict = construct_heb_edges(files_path, start_date, end_date, non_parliamentarians_nodes)
G = construct_heb_network(edge_dict)
# print(G.edges(data=True))
# Visualize the graph
visualize_graph(G)