from Mutilayer_net_community_detect_model import MutillayerNetCommunityDetect
from GenerateRandomNet import Multilayer_random_network
from utils import *

if __name__ == '__main__':
    # generate random network
    MRN = Multilayer_random_network()
    layers, G, G_Matrix = MRN.generate_mulit_net()
    show_mx_matrix_graph(G, layers, [G.nodes])

    # import model,communites
    model = MutillayerNetCommunityDetect(G_Matrix, 9)
    model.__init__(G_Matrix, 9)
    for t in range(1700):
        model.iterate()
        while not model.has_converged():
            model.iterate()
    communities_dict = model.sets_of_community_nodes()
    # 按照编号顺序将社区排号
    communities_dict = dict(sorted(communities_dict.items(), key=lambda item: item[0]))
    # 按社区编号将社区节点粒子排序,这样每一次画图的时候每个社区颜色不会改变
    communities_list = [value for value in communities_dict.values()]
    show_mx_matrix_graph(G, layers, communities_list)





