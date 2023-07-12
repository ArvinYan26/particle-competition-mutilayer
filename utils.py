import matplotlib.pyplot as plt
import networkx as nx
import multinetx as mx
import random

def generate_different_colors(num_of_colors):
    # 随机生成9种不同的颜色，并存储在list中
    color_list = []
    for i in range(num_of_colors):
        # 生成随机RGB颜色
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # 转换为16进制格式
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        color_list.append(hex_color)
    return color_list

def show_mx_matrix_graph(G, layers, communities_nodes):
    """
    showo the graph and it's Matrix
    @param G: the Graph of the multilayers communities networks
    @param layers: the graph of every layers
    @param communities_nodes: the nodes of communities
    @return:
    """
    pos = mx.get_position(G, mx.fruchterman_reingold_layout(layers[0], seed=1),
                          layer_vertical_shift=1.4, layer_horizontal_shift=0.0, proj_angle=7)
    if len(communities_nodes) == 1:
        plt.figure(figsize=(20, 20))
        plt.imshow(mx.adjacency_matrix(G).todense(),
                   origin='upper', interpolation='nearest', cmap=plt.cm.jet_r)
        # show the graph
        plt.figure(figsize=(20, 15))
        mx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5)
        mx.draw_networkx_nodes(G, pos, node_size=300, node_color='r')
    else:
        # 定义节点颜色和形状字典
        color_list = generate_different_colors(len(communities_nodes))
        community_styles = {}
        for i, comm in enumerate(communities_nodes):
            shape_list = ['o', 's', '^', 'd', 'D', 'h', 'v', 'p', '1']
            community_styles[i] = {'color': color_list[i], 'shape':shape_list[i]}
        # 绘制图形
        plt.figure(figsize=(20, 15))
        for i, comm in enumerate(communities_nodes):
            mx.draw_networkx_nodes(G, pos, nodelist=comm, node_size=300,
                                   node_shape=community_styles[i]['shape'],
                                   node_color=community_styles[i]['color'])
        mx.draw_networkx_edges(G, pos, alpha=0.2)
        # 添加图例
        handles = []
        labels = []
        for i, comm in enumerate(communities_nodes):
            label = f'Community {i + 1}'
            h = plt.scatter([], [], s=300, marker=community_styles[i]['shape'], color=community_styles[i]['color'])
            handles.append(h)
            labels.append(label)
        plt.legend(handles, labels, scatterpoints=1, fontsize=14)
    plt.show()
