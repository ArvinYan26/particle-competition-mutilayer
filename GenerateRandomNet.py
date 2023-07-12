import numpy as np
import math
from utils import *
import multinetx as mx

class Multilayer_random_network:
    def __init__(self):
        self.NETWORK_SIZE = 48    #单层网络节点数
        # PROBABILITY_OF_EAGE=0.8  #Limited to global
        self.pout = 0.05
        self.pin = 0.8
        self.zin = 0
        self.zout = 0
        self.K = 17
        self.M = 3  # Community Number
        self.layers_num = 3

    def GenerateAdjMatrix(self, NETWORK_SIZE):
        Amatrix = [[0 for i in range(NETWORK_SIZE)] for i in range(NETWORK_SIZE)]
        # def generateRandomNetwork()閿?
        for i in range(0, NETWORK_SIZE):
            for j in range(i, NETWORK_SIZE):
                Amatrix[i][j] = Amatrix[j][i] = 0
        return Amatrix

    def CommunityStructuredNetwork(self, Amatrix):
        zin = 0
        zout = 0
        ztotal = 0
        NETWORK_SIZE = len(Amatrix)          # ==== 48
        # intvl和后边topcommunity里面的不一样
        self.single_intvl = int(NETWORK_SIZE / self.M)   # ==== 16

        # INTRAcommunity
        while (ztotal * 2 / NETWORK_SIZE < self.K):
            # intra-community
            if (zin * 2 / NETWORK_SIZE < 15):
                for m in range(self.M):
                    v1 = random.randint(m * self.single_intvl, (m + 1) * self.single_intvl - 1)
                    v2 = random.randint(m * self.single_intvl, (m + 1) * self.single_intvl - 1)

                    if (v1 == v2):
                        continue
                    # if (zin/NETWORK_SIZE< K):
                    probability = np.random.random()
                    if (probability <= self.pin):
                        if (Amatrix[v1][v2] == 0):
                            Amatrix[v1][v2] = Amatrix[v2][v1] = 1
                            zin = zin + 1

            # inter-community
            if (zout * 2 / NETWORK_SIZE < 2):
                keepWalking = True
                while (keepWalking):
                    v3 = random.randint(0, NETWORK_SIZE - 1)
                    v4 = random.randint(0, NETWORK_SIZE - 1)
                    if (math.floor(v3 / self.single_intvl) != math.floor(v4 / self.single_intvl)):  # Same community?
                        keepWalking = False  # If so, move on (If not, choose new 2-random nodes: keep on While)
                probability = np.random.random()
                if (probability <= self.pout):
                    if (Amatrix[v3][v4] == 0):
                        Amatrix[v3][v4] = Amatrix[v4][v3] = 1
                        zout = zout + 1
            ztotal = zin + zout

        return Amatrix

    def generate_mulit_net(self):
        # self = Multilayer_random_network()
        # 存储生成的单层网络邻接矩阵
        MatrixList = []
        # layers：存储每一层社区网络
        layers = []
        for i in range(self.layers_num):
            # 生成三个同层社区网络（三层社区网络中的一层），每层是三个社区结构，
            Amatrix = self.GenerateAdjMatrix(self.NETWORK_SIZE)
            # 将邻接矩阵转化为只有0,1的邻接矩阵
            Amatrix = self.CommunityStructuredNetwork(Amatrix)
            # 将每一个邻接矩阵都画出来
            MatrixList.append(Amatrix)
            G = nx.from_numpy_matrix(np.array(Amatrix))
            layers.append(G)
        adj_block = mx.lil_matrix(np.zeros((self.NETWORK_SIZE * 3, self.NETWORK_SIZE * 3)))
        adj_block[0:  self.NETWORK_SIZE, self.NETWORK_SIZE:2 * self.NETWORK_SIZE] = np.identity(self.NETWORK_SIZE)  # L_12
        adj_block[0:  self.NETWORK_SIZE, 2 * self.NETWORK_SIZE:3 * self.NETWORK_SIZE] = np.identity(self.NETWORK_SIZE)  # L_13
        adj_block[self.NETWORK_SIZE:2 * self.NETWORK_SIZE, 2 * self.NETWORK_SIZE:3 * self.NETWORK_SIZE] = np.identity(
            self.NETWORK_SIZE)  # L_23

        adj_block += adj_block.T
        mg = mx.MultilayerGraph(list_of_layers=layers,
                                inter_adjacency_matrix=adj_block)
        multi_network_matrix = nx.to_numpy_matrix(mg)
        multi_network_matrix = np.asarray(multi_network_matrix)

        return layers, mg, multi_network_matrix

