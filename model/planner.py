from opt import adadelta
from util.misc import *

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import numpy as np
import yaml

N_HIDDEN = 300
THINK_TIME = 10

OPT_PARAMS = Struct(**yaml.load("""
    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10
"""))

class Planner(object):
    def __init__(self, categorical=False):
        self.net = ApolloNet()
        self.opt_state = adadelta.State()
        self.categorical = categorical

    def forward(self, data, train=False):
        features = np.asarray([d.features for d in data])
        max_len = max(len(d.demonstration) for d in data)
        if self.categorical:
            n_targets = 1000
            targets = np.zeros((len(data), max_len))
        else:
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

        l_plan = self.think(l_relu_repr, randomize=train)

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
        else:
            ll_targets = None
            ll_masks = None

        loss, ll_predictions = self.act(l_plan, max_len, data, ll_targets,
                ll_masks, self_init=not train)

        if train:
            self.net.backward()
            adadelta.update(self.net, self.opt_state, OPT_PARAMS)

        return loss, ll_predictions

    def lstm(self, prefix, inputs):
        l_seed = "seed_%s" % prefix
        lt_hidden = "hidden_%s_%%d" % prefix
        lt_mem = "mem_%s_%%d" % prefix
        lt_concat = "concat_%s_%%d" % prefix
        lt_lstm = "lstm_%s_%%d" % prefix

        p_lstm = [p % prefix for p in ("lstm_i_val_%s", "lstm_i_gate_%s",
                "lstm_f_gate_%s", "lstm_o_gate_%s")]

        batch_size = self.net.blobs[inputs[0][0]].shape[0]
        self.net.f(NumpyData(l_seed, np.zeros((batch_size, N_HIDDEN))))
        l_prev_hidden = l_seed
        l_prev_mem = l_seed
        for t in range(len(inputs)):
            l_hidden = lt_hidden % t
            l_mem = lt_mem % t
            l_concat = lt_concat % t
            l_lstm = lt_lstm % t

            self.net.f(Concat(l_concat, bottoms=[l_prev_hidden]+inputs[t]))
            self.net.f(LstmUnit(l_lstm, bottoms=[l_concat, l_prev_mem],
                    param_names=p_lstm, tops=[l_hidden, l_mem],
                    num_cells=N_HIDDEN))

            l_prev_hidden = l_hidden
            l_prev_mem = l_mem
            yield l_hidden

    def think(self, l_repr, randomize):
        time = np.random.randint(THINK_TIME) + 1 if randomize else THINK_TIME
        reprs = [l for l in self.lstm("think", [[l_repr] for i in
                range(time)])]
        return reprs[-1]

    def act(self, l_plan, max_len, data, ll_targets, ll_masks, self_init):
        #n_actions = data[0].n_actions
        if self.categorical:
            n_targets = 1000
        else:
            n_targets = len(data[0].demonstration[0])

        lt_state_repr = "state_repr_%d"
        lt_pred = "pred_%d"
        lt_apply_mask = "apply_mask_%d"
        lt_loss = "loss_%d"
        ll_state_reprs = [lt_state_repr % t for t in range(max_len)]

        init_state_reprs = np.asarray([d.inject_state_features(d.init) 
                for d in data])
        self.net.f(NumpyData(ll_state_reprs[0], init_state_reprs))

        loss = 0 if ll_targets is not None else None
        ll_predictions = []
        for t, l_hidden in enumerate(self.lstm("act", [[l] for l in
                ll_state_reprs[:-1]])):
            l_pred = lt_pred % t
            l_apply_mask = lt_apply_mask % t
            l_loss = lt_loss % t

            self.net.f(InnerProduct(l_pred, n_targets, bottoms=[l_hidden]))
            ll_predictions.append(l_pred)

            if ll_targets is not None:
                l_mask = ll_masks[t]
                l_target = ll_targets[t]

                if self.categorical:
                    loss += self.net.f(SoftmaxWithLoss(l_loss, 
                            bottoms=[l_pred, l_target]))
                else:
                    self.net.f(Eltwise(l_apply_mask, "PROD", 
                            bottoms=[l_pred, l_mask]))
                    loss += self.net.f(EuclideanLoss(l_loss, 
                            bottoms=[l_apply_mask, l_target]))

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

        return loss, ll_predictions

    def demonstrate(self, data):
        loss, _ = self.forward(data, train=True)
        return loss

    def predict(self, data):
        _, ll_predictions = self.forward(data, train=False)
        predictions = []
        for i_datum, datum in enumerate(data):
            prediction = [datum.init]
            for t in range(len(datum.demonstration) - 1):
                #state = self.net.blobs[ll_predictions[t]].data[i_datum, :].argmax() - 1
                state = self.net.blobs[ll_predictions[t]].data[i_datum, :]
                prediction.append(tuple(state))
            predictions.append(tuple(prediction))
        return predictions
