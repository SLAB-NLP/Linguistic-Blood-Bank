from pyvis.network import Network


def set_node_color(feat_color_dict, feat_df, feat, node, feat_idx2value):
    if feat == 'No feature':
        node_color = '#f86368'
    else:
        if node in feat_color_dict:
            node_color = feat_color_dict[node]
        else:
            node_color = feat_color_dict[feat_idx2value[feat][int(feat_df[feat][node])]]
    return node_color


def set_edge_color(feat_color_dict, feat_df, feat, node, feat_idx2value):
    if feat == 'No feature':
        node_color = '#787c82'
    else:
        if node in feat_color_dict:
            node_color = feat_color_dict[node]
        else:
            node_color = feat_color_dict[feat_idx2value[feat][int(feat_df[feat][node])]]
    return node_color


def set_node_value(feat_df, feat, node, feat_idx2value, name2wiki):
    if feat == 'No feature':
        node_value = name2wiki[node]
    else:
        if node not in feat_df.index:
            return node
        if feat_df[feat][node] == -1:
            return 'None'
        node_value = feat_idx2value[feat][int(feat_df[feat][node])]
    return str(node_value)


def get_graph(edge_weight_amplifier, feat, feat_df, feat_idx2value, filtered_df, mrr_tr, name2wiki, selected_langs,
              val_colors_dict):
    dynamic_graph = Network(height='1080px', width="1250px", bgcolor='white', font_color='black', directed=True)
    node_color = '#f86368'
    for i, col in enumerate(filtered_df.columns):
        for j, row in enumerate(filtered_df.index):
            if col in selected_langs:
                dynamic_graph.add_node(col, color=set_node_color(val_colors_dict, feat_df, feat, col, feat_idx2value),
                                       title=set_node_value(feat_df, feat, col, feat_idx2value, name2wiki),
                                       value=filtered_df[col][col])  # set color here
            else:
                continue  # important so no edge would be formed
            if i != j and (mrr_tr[0] < filtered_df[col][row] < mrr_tr[1]) and row in selected_langs:
                dynamic_graph.add_node(row, color=set_node_color(val_colors_dict, feat_df, feat, row, feat_idx2value),
                                       title=set_node_value(feat_df, feat, row, feat_idx2value, name2wiki),
                                       value=filtered_df[row][row])  # set color here
                dynamic_graph.add_edge(row, col, width=filtered_df[col][row] * edge_weight_amplifier,
                                       color=set_edge_color(val_colors_dict, feat_df, feat, row, feat_idx2value),
                                       label=f"{filtered_df[col][row]:.3f}")
    dynamic_graph.set_edge_smooth("dynamic")
    # Generate network with specific layout settings
    dynamic_graph.repulsion(node_distance=120, central_gravity=0.3,
                            spring_length=110, spring_strength=0.000003,
                            damping=0.95)
    return dynamic_graph
