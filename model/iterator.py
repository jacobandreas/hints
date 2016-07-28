from model.components import lstm, bilstm
from opt import adadelta
from util.misc import *

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import numpy as np
import yaml

N_HIDDEN = 300

OPT_PARAMS = Struct(**yaml.load("""
    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10
"""))

class Iterator(object):
    def __init__(self, categorical=False):
        self.net = ApolloNet()
        self.opt_state = adadelta.State()
        assert not categorical

    def forward(self, data, train=False):
        features = np.asarray([d.features for d in data])
        max_len = max(len(d.demonstration) for d in data)

        n_targets = len(d.demonstration[0])
        targets = np.zeros((len(data), max_len, n_targets))
        masks = np.zeros((len(data), max_len, n_targets))

        for i_datum in range(len(data)):
            demo_len = len(data[i_datum].demonstration)
            targets[i_datum, :demo_len, ...] = data[i_datum].demonstration
            masks[i_datum, :demo_len, ...] = 1

        l_features = "features"
        l_ip_repr = "ip_repr"
        l_relu_repr = "relu_repr"
        lt_mask = "mask_%d"
        lt_target = "target_%d"

        self.net.clear_forward()
        self.net.f(NumpyData(l_features, features))
        self.net.f(InnerProduct(l_ip_repr, N_HIDDEN, bottoms=[l_features]))
        self.net.f(ReLU(l_relu_repr, bottoms=[l_ip_repr]))

        ll_pred1 = self.initialize(l_relu_repr, max_len, n_targets, data,
                self_init=not train)
        ll_pred2 = self.refine(ll_pred1, n_targets)
        #ll_pred2 = ll_pred1

        if train:
            ll_targets = []
            ll_masks = []
            for i_target in range(1, max_len):
                l_target = lt_target % i_target
                l_mask = lt_mask % i_target
                self.net.f(NumpyData(l_target, targets[:, i_target]))
                self.net.f(NumpyData(l_mask, masks[:, i_target]))
                ll_targets.append(l_target)
                ll_masks.append(l_mask)

            loss1 = self.loss("pred1", ll_pred1, ll_targets, ll_masks)
            loss2 = self.loss("pred2", ll_pred2, ll_targets, ll_masks)
            loss = np.asarray([loss1, loss2])
            self.net.backward()
            adadelta.update(self.net, self.opt_state, OPT_PARAMS)
        else:
            loss = None

        return loss, ll_pred2

    def initialize(self, l_repr, max_len, n_targets, data, self_init=False):
        lt_state_repr = "state_repr_%d"
        lt_pred = "pred_%d"

        ll_state_reprs = [lt_state_repr % t for t in range(max_len)]

        init_state_reprs = np.asarray([d.inject_state_features(d.init) 
                for d in data])
        self.net.f(NumpyData(ll_state_reprs[0], init_state_reprs))

        ll_predictions = []

        for t, l_hidden in enumerate(
                lstm("init", [[l] for l in ll_state_reprs[:-1]], N_HIDDEN, self.net)):
            l_pred = lt_pred % t

            self.net.f(InnerProduct(l_pred, n_targets, bottoms=[l_hidden]))
            ll_predictions.append(l_pred)

            if self_init:
                state_reprs = []
                for i_datum, datum in enumerate(data):
                    state = self.net.blobs[l_pred].data[i_datum, :]
                    state = np.round(state).astype(int)
                    state_reprs.append(datum.inject_state_features(state))
            else:
                state_reprs = []
                for i_datum, datum in enumerate(data):
                    if t < len(datum.demonstration) - 1:
                        state_reprs.append(datum.inject_state_features(datum.demonstration[t+1]))
                    else:
                        state_reprs.append(np.zeros(self.net.blobs[ll_state_reprs[0]].shape[1:]))
            self.net.f(NumpyData(ll_state_reprs[t+1], np.asarray(state_reprs)))

        return ll_predictions

    def refine(self, ll_prev, n_targets):
        ll_hidden = bilstm("refine", [[l] for l in ll_prev], N_HIDDEN, self.net)
        lt_concat = "refine_concat_%d"
        lt_pred = "refine_pred_%d"
        ll_predictions = []
        for t, l_hidden in enumerate(ll_hidden):
            l_concat = lt_concat % t
            l_pred = lt_pred % t
            #self.net.f(Concat(l_concat, bottoms=[, l_hidden]))
            self.net.f(InnerProduct(l_pred, n_targets, bottoms=[l_hidden]))
            ll_predictions.append(l_pred)
        return ll_predictions

    def loss(self, prefix, ll_pred, ll_targets, ll_masks):
        lt_apply_mask = "apply_mask_%%d_%s" % prefix
        lt_loss = "loss_%%d_%s" % prefix
        loss = 0
        for t, l_pred in enumerate(ll_pred):
            l_pred = ll_pred[t]
            l_apply_mask = lt_apply_mask % t
            l_loss = lt_loss % t

            l_mask = ll_masks[t]
            l_target = ll_targets[t]

            self.net.f(Eltwise(l_apply_mask, "PROD", 
                    bottoms=[l_pred, l_mask]))
            loss += self.net.f(EuclideanLoss(l_loss, 
                    bottoms=[l_apply_mask, l_target]))
        return loss

    def demonstrate(self, data):
        loss, _ = self.forward(data, train=True)
        return loss

    def predict(self, data):
        _, ll_predictions = self.forward(data, train=False)
        predictions = []
        for i_datum, datum in enumerate(data):
            prediction = [datum.init]
            for t in range(len(datum.demonstration) - 1):
                state = self.net.blobs[ll_predictions[t]].data[i_datum, :]
                prediction.append(tuple(state))
            predictions.append(tuple(prediction))
        return predictions
