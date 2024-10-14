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

        # Create a dictionary to store the maximum mask value for each undirected edge
        edge_importance = {}

        # Fill the edge_importance dictionary with the maximum values for (u, v) and (v, u)
        expl_mask = expl_masks_split[i]
        for idx, (u, v) in enumerate(edges):
            # Sort the edge to treat (u, v) and (v, u) the same
            edge_key = tuple(sorted((u, v)))
            if edge_key not in edge_importance:
                edge_importance[edge_key] = expl_mask[idx]
            else:
                # Take the maximum mask value for the undirected edge
                edge_importance[edge_key] = max(edge_importance[edge_key], expl_mask[idx])

        # Convert edge_importance dict to lists for plotting
        all_edges = list(edge_importance.keys())
        all_importances = list(edge_importance.values())

        # Sort edges by importance to get top-k
        sorted_indices = sorted(range(len(all_importances)), key=lambda k: all_importances[k], reverse=True)
        top_k_indices = sorted_indices[:top_k]

        top_k_edges = [all_edges[idx] for idx in top_k_indices]
        top_k_importances = [all_importances[idx] for idx in top_k_indices]

        # Non-top-k edges
        non_top_k_edges = [all_edges[idx] for idx in sorted_indices[top_k:]]
        
        # Create a spring layout for consistent node positions
        pos = nx.spring_layout(G)

        # Set up subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Draw nodes
        node_options = {
            'node_size': 20,
            'node_color': 'lightblue',
            'alpha': 0.9
        }

        # Plot 1: Entire Graph (Black and Red Edges)
        nx.draw_networkx_nodes(G, pos, ax=axs[0], **node_options)
        # Color edges: red for top-k, black for the rest
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color=['red' if edge in top_k_edges else 'black' for edge in all_edges], ax=axs[0])
        axs[0].set_title(f'Graph {i+1}: Full (Black + Red)', fontsize=16)
        axs[0].axis('off')

        # Plot 2: Only Non-Top K Edges (Black Edges)
        nx.draw_networkx_nodes(G, pos, ax=axs[1], **node_options)
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color=['black' if edge in non_top_k_edges else 'lightgray' for edge in all_edges], ax=axs[1])
        axs[1].set_title(f'Graph {i+1}: Black Edges', fontsize=16)
        axs[1].axis('off')

        # Plot 3: Only Top K Important Edges (Red Edges)
        nx.draw_networkx_nodes(G, pos, ax=axs[2], **node_options)
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color=['red' if edge in top_k_edges else 'lightgray' for edge in all_edges], edge_cmap=plt.cm.Reds, ax=axs[2])
        axs[2].set_title(f'Graph {i+1}: Red Edges', fontsize=16)
        axs[2].axis('off')

        # Show the subplots
        plt.tight_layout()
        plt.show()

def visualize_pred_vs_gt(graphs, expl_masks_split, edge_label, top_k=10):

    for i, graph_data in enumerate(graphs):
        # Create a NetworkX graph
        G_pred = nx.Graph()

        # Add nodes
        num_nodes = graph_data['nodes'].shape[0]
        G_pred.add_nodes_from(range(num_nodes))

        # Add edges
        edges = graph_data['edges'].t().tolist()
        G_pred.add_edges_from(edges)

        # Create a dictionary to store the maximum mask value for each undirected edge
        edge_importance = {}

        # Fill the edge_importance dictionary with the maximum values for (u, v) and (v, u)
        expl_mask = expl_masks_split[i]
        for idx, (u, v) in enumerate(edges):
            # Sort the edge to treat (u, v) and (v, u) the same
            edge_key = tuple(sorted((u, v)))
            if edge_key not in edge_importance:
                edge_importance[edge_key] = expl_mask[idx]
            else:
                # Take the maximum mask value for the undirected edge
                edge_importance[edge_key] = max(edge_importance[edge_key], expl_mask[idx])

        # Convert edge_importance dict to lists for plotting
        all_edges = list(edge_importance.keys())
        all_importances = list(edge_importance.values())

        # Sort edges by importance to get top-k
        sorted_indices = sorted(range(len(all_importances)), key=lambda k: all_importances[k], reverse=True)
        top_k_indices = sorted_indices[:top_k]

        top_k_edges = [all_edges[idx] for idx in top_k_indices]
        top_k_importances = [all_importances[idx] for idx in top_k_indices]

        # Non-top-k edges
        non_top_k_edges = [all_edges[idx] for idx in sorted_indices[top_k:]]
        
        # Create a spring layout for consistent node positions
        pos = nx.spring_layout(G_pred)

        # Set up subplots
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        # Draw nodes
        node_options = {
            'node_size': 20,
            'node_color': 'lightblue',
            'alpha': 0.9
        }

        # Plot 1: Entire Graph (Black and Red Edges)
        nx.draw_networkx_nodes(G_pred, pos, ax=axs[0], **node_options)
        # Color edges: red for top-k, black for the rest
        nx.draw_networkx_edges(G_pred, pos, edgelist=all_edges, edge_color=['red' if edge in top_k_edges else 'black' for edge in all_edges], ax=axs[0])
        axs[0].set_title(f'Graph {i+1}: Full (Black + Red)', fontsize=16)
        axs[0].axis('off')

        # Plot 2: Only Non-Top K Edges (Black Edges)
        nx.draw_networkx_nodes(G_pred, pos, ax=axs[1], **node_options)
        nx.draw_networkx_edges(G_pred, pos, edgelist=all_edges, edge_color=['black' if edge in non_top_k_edges else 'lightgray' for edge in all_edges], ax=axs[1])
        axs[1].set_title(f'Graph {i+1}: Black Edges', fontsize=16)
        axs[1].axis('off')

        # Plot 3: Only Top K Important Edges (Red Edges)
        nx.draw_networkx_nodes(G_pred, pos, ax=axs[2], **node_options)
        nx.draw_networkx_edges(G_pred, pos, edgelist=all_edges, edge_color=['red' if edge in top_k_edges else 'lightgray' for edge in all_edges], edge_cmap=plt.cm.Reds, ax=axs[2])
        axs[2].set_title(f'Graph {i+1}: Red Edges', fontsize=16)
        axs[2].axis('off')

        # Plot 4: Ground Truth
        G_gt = nx.Graph()
        
        # Add nodes
        G_gt.add_nodes_from(range(num_nodes))

        # Add ground truth edges
        for idx, (u, v) in enumerate(edges):
            if edge_label[idx]:  # Only add edges that are labeled as true
                G_gt.add_edge(u, v)

        # Draw the predicted graph
        nx.draw_networkx_nodes(G_pred, pos, ax=axs[3], **node_options)
        # Draw predicted edges in black and ground truth edges in green
        nx.draw_networkx_edges(G_pred, pos, ax=axs[3], edge_color='black')
        nx.draw_networkx_edges(G_gt, pos, ax=axs[3], edge_color='green')
        axs[3].set_title(f'Graph {i+1}: Full Graph with Ground Truth', fontsize=16)
        axs[3].axis('off')
        axs[3].legend()

        # Show the subplots
        plt.tight_layout()
        plt.show()


def visualize_cfs(original_adj, counterfactual_adj):

    original_graph = nx.from_numpy_array(original_adj.cpu().detach().numpy())
    counterfactual_graph = nx.from_numpy_array(counterfactual_adj.cpu().detach().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    pos = nx.spring_layout(original_graph)
    nx.draw(original_graph, pos, ax=axes[0], with_labels=False, node_color='lightblue', edge_color='gray', node_size=20)
    axes[0].set_title('Original Graph')

    nx.draw(counterfactual_graph, pos, ax=axes[1], with_labels=False, node_color='lightgreen', edge_color='gray', node_size=20)
    axes[1].set_title('Counterfactual Graph')

    plt.tight_layout()
    plt.show()
