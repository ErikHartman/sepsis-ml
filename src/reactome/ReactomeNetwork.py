
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import re

""" 
Original file from https://github.com/marakeby/pnet_prostate_paper/blob/master/data/pathways/reactome.py
Modified to fit our dir.

"""

def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_levels=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_levels:
            diff = n_levels - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers

class Reactome():
    def __init__(self, path_file, protein_file):
        self.hierarchy = pd.read_csv(path_file, names=['parent','child'], sep="\t")
        self.proteins = pd.read_csv(protein_file, names = ['UniProt_id', 'Reactome_id', 'URL', 'Description','Evidence Code','Species'], sep="\t")

class ReactomeNetwork():

    def __init__(self,  all_paths_file = "data/reactome/ReactomePathwaysRelation.txt", protein_file = "data/reactome/UniProt2Reactome.txt"):
        self.reactome = Reactome(all_paths_file, protein_file) 
        self.netx = self.get_reactome_networkx()
        
    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'parent','child', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):
        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_levels=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_levels=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (protein level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find proteins belonging to these pathways
        protein_df = self.reactome.proteins

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            proteins = protein_df[protein_df['Reactome_id'] == pathway_name]['UniProt_id'].unique()
            if len(proteins) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = proteins

        layers.append(dict)
        return layers


if __name__ == '__main__':
    RN = ReactomeNetwork()
    print(RN.info())
    