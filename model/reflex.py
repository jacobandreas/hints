from opt import adadelta
from util.misc import *

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import numpy as np
import yaml

N_HIDDEN = 100
N_LAYERS = 2

OPT_PARAMS = Struct(**yaml.load("""
    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10
"""))

class Reflex(object):
    def __init__(self):
        self.net = ApolloNet()
        self.opt_state = adadelta.State()
        self.n_targets = 2

    def forward(self, features, positions, targets, masks, train=False):
        features = np.asarray(features)
        positions = np.asarray(positions)
        target = np.asarray(targets)
        mask = np.asarray(masks)

        l_features = "features"
        l_positions = "positions"
        l_concat = "concat"
        lt_ip = "ip_%d"
        lt_relu = "relu_%d"
        l_target = "targets"
        l_mask = "mask"
        l_mul_mask = "mul_mask"
        l_loss = "loss"

        self.net.clear_forward()
        self.net.f(NumpyData(l_features, features))
        self.net.f(NumpyData(l_positions, positions))
        self.net.f(Concat(l_concat, bottoms=[l_features, l_positions]))

        l_prev = l_concat
        for i_layer in range(N_LAYERS - 1):
            l_ip = lt_ip % i_layer
            l_relu = lt_relu % i_layer
            self.net.f(InnerProduct(l_ip, N_HIDDEN, bottoms=[l_prev]))
            self.net.f(ReLU(l_relu, bottoms=[l_ip]))
            l_prev = l_relu

        l_ip = lt_ip % (N_LAYERS - 1)
        self.net.f(InnerProduct(l_ip, self.n_targets, bottoms=[l_prev]))
        self.l_predict = l_ip

        if train:
            self.net.f(NumpyData(l_mask, mask))
            self.net.f(Eltwise(l_mul_mask, "PROD", bottoms=[l_mask, self.l_predict]))
            self.net.f(NumpyData(l_target, target))
            loss = self.net.f(EuclideanLoss(l_loss, bottoms=[l_target, l_mul_mask]))
            self.net.backward()
            adadelta.update(self.net, self.opt_state, OPT_PARAMS)
            return np.asarray([loss])


    def demonstrate(self, data):
        max_len = max(len(d.demonstration) for d in data)
        loss = 0
        for t in range(1, max_len):
            features = []
            positions = []
            targets = []
            masks = []

            for datum in data:
                features.append(datum.features)
                if t < len(datum.demonstration):
                    positions.append(datum.demonstration[t-1])
                    targets.append(datum.demonstration[t])
                    masks.append((1,) * self.n_targets)
                else:
                    positions.append((0,) * self.n_targets)
                    targets.append((0,) * self.n_targets)
                    masks.append((0,) * self.n_targets)

            loss += self.forward(features, positions, targets, masks,
                    train=True)

        return loss

    def predict(self, data):
        max_len = max(len(d.demonstration) for d in data)
        print max_len
        paths = [[datum.init] for datum in data]
        for t in range(1, max_len):
            features = []
            positions = []

            for i_datum in range(len(data)):
                features.append(datum.features)
                positions.append(paths[i_datum][-1])
            
            self.forward(features, positions, [], [])

            for i_datum in range(len(data)):
                paths[i_datum].append(tuple(self.net.blobs[self.l_predict].data[i_datum]))

        return paths
