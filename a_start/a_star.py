import osmnx as ox
import networkx as nx

# Carrega o grafo da região desejada
lugar = "Paraná, Toledo"
G = ox.graph_from_place(lugar, network_type="drive")

# Define os nós de origem e destino
origem = list(G.nodes)[0]  # Ajuste com um nó específico ou coordenadas desejadas
destino = list(G.nodes)[-1]  


caminho = nx.astar_path(G, origem, destino, heuristic=nx.shortest_path_length)

fig, ax = ox.plot_graph_route(G, caminho, route_linewidth=2, node_size=0, bgcolor="w")