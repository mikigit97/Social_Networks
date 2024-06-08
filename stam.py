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
colormap = plt.colormaps.get_cmap('tab20')
colors = colormap(range(20))
# print(colors)
i = 0
for community in range(5):
    # color = next(itertools.cycle(colors))
    # print(color)
    i+=1
    color = colors[i]
    print(color)