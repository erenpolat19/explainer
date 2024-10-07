import networkx as nx
import matplotlib.pyplot as plt

def extract_individual_graphs(batch, expl_mask):
    # Get the number of graphs in the batch
    num_graphs = batch.batch.max().item() + 1

    graphs = []
    expl_masks_split = [] 
    
    # Iterate over each graph in the batch
    for graph_idx in range(num_graphs):
        # Find nodes that belong to the current graph
        node_mask = batch.batch == graph_idx
        
        # Get node indices for this graph
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)
        
        # Extract node features for this graph
        node_features = batch.x[node_indices]
        
        # Filter edges for the current graph
        edge_mask = node_mask[batch.edge_index[0]]  # Only keep edges where the source node belongs to the current graph
        edge_index = batch.edge_index[:, edge_mask]
        
        # Remap the node indices for the edge_index (so indices are zero-based for each graph)
        edge_index_remapped = edge_index.clone()
        node_id_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Apply remapping safely with default if the node is missing
        edge_index_remapped = edge_index_remapped.apply_(lambda idx: node_id_map.get(idx, -1))

        # print(f"Graph {graph_idx}: node_indices = {node_indices.tolist()}")
        # print(f"Graph {graph_idx}: edge_index (before remapping) = {edge_index}")
        # print(f"Graph {graph_idx}: edge_index_remapped (after remapping) = {edge_index_remapped}")

        # Remove edges with unmapped nodes (which were not part of the node_id_map)
        valid_edges_mask = (edge_index_remapped[0] != -1) & (edge_index_remapped[1] != -1)
        edge_index_remapped = edge_index_remapped[:, valid_edges_mask]

        # Filter expl_mask for the current graph's edges
        expl_mask_split = expl_mask[edge_mask][valid_edges_mask]
        expl_masks_split.append(expl_mask_split)

        # Create a dictionary for the current graph
        graph_data = {
            'nodes': node_features,
            'edges': edge_index_remapped
        }
        
        graphs.append(graph_data)

    return graphs, expl_masks_split

def visualize(graphs, expl_masks_split, top_k):
    for i, graph_data in enumerate(graphs):
        # Create a NetworkX graph
        G = nx.Graph()

        # Add nodes
        num_nodes = graph_data['nodes'].shape[0]
        G.add_nodes_from(range(num_nodes))

        # Add edges
        edges = graph_data['edges'].t().tolist()
        G.add_edges_from(edges)

        # Set up the plot
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G) 
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightblue", alpha=0.9)

        # Draw edges
        edge_colors = 'black'
        expl_mask = expl_masks_split[i]  # Use the explanation mask corresponding to this graph
        sorted_indices = expl_mask.argsort(descending=True)
        top_k_indices = sorted_indices[:top_k]  # Get indices of top k edges
        top_k_edges = [edges[idx] for idx in top_k_indices]
        top_k_mask = expl_mask[top_k_indices]  # Corresponding importance values

        non_top_k_indices = sorted_indices[top_k:]
        non_top_k_edges = [edges[idx] for idx in non_top_k_indices]

        # Add the top k edges to the graph
        G.add_edges_from(top_k_edges)
        edge_colors = plt.cm.Reds(top_k_mask) 
        nx.draw_networkx_edges(G, pos, edgelist=non_top_k_edges, edge_color='black')
        nx.draw_networkx_edges(G, pos, edgelist=top_k_edges, edge_color='red')
        
        # Title
        plt.title(f'Graph {i+1}', fontsize=16)
        plt.axis('off')  

        # Show the plot
        plt.show()