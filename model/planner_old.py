from opt import adadelta
from util import *

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import numpy as np
import yaml

N_HIDDEN = 100

OPT_PARAMS = Struct(**yaml.load("""
    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10
"""))

class Planner(object):
    def __init__(self):
        self.net = ApolloNet()
        self.n_iters = 2
        self.anytime = False
        #self.do_forward = True
        #self.do_backward = True
        self.opt_state = adadelta.State()

    def forward(self, data, train=False):
        l_input = "features"
        lt_target = "target_%d"
        lt_mask = "mask_%d"

        features = np.asarray([d.features for d in data])

        max_len = max(len(d.demonstration) for d in data)
        targets = np.zeros((len(data), max_len, len(data[0].demonstration[0])))
        masks = np.ones(targets.shape)
        for i_datum in range(len(data)):
            demo_len = len(data[i_datum].demonstration)
            targets[i_datum, :demo_len, :] = data[i_datum].demonstration
            masks[i_datum, demo_len:, :] = 0

        self.net.clear_forward()
        self.net.f(NumpyData(l_input, features))
        self.net.f(InnerProduct("l_ip_features", N_HIDDEN, bottoms=[l_input]))
        self.net.f(ReLU("l_relu_features", bottoms=["l_ip_features"]))
        l_features = "l_relu_features"

        ll_targets = []
        ll_masks = []
        n_waypoints = targets.shape[1]
        for i_tgt in range(n_waypoints):
            l_target = lt_target % i_tgt
            l_mask = lt_mask % i_tgt
            self.net.f(NumpyData(l_target, targets[:, i_tgt, :]))
            self.net.f(NumpyData(l_mask, masks[:, i_tgt, :]))
            ll_targets.append(l_target)
            ll_masks.append(l_mask)

        losses = np.zeros(self.n_iters)
        for i_iter in range(self.n_iters):
            losses[i_iter] = self.iter(i_iter, l_features, ll_targets, ll_masks,
                    n_waypoints, data, self_init=not train)

        if train:
            self.net.backward()
            adadelta.update(self.net, self.opt_state, OPT_PARAMS)

        return losses

    def iter(self, i_iter, l_features, ll_targets, ll_masks, n_waypoints, data, self_init=True):
        n_batch = self.net.blobs[l_features].shape[0]
        l_seed = "seed_%d" % i_iter
        lt_hidden = "hidden_%s_%d_%d"
        lt_mem = "mem_%s_%d_%d"
        lt_concat1 = "concat1_%s_%d_%d"
        lt_lstm = "lstm_%s_%d_%d"
        lt_ip = "ip_%d_%d"
        lt_relu = "relu_%d_%d"
        lt_concat2 = "concat2_%d_%d"
        lt_predict = "predict_%d_%d"
        lt_mul_mask = "mul_mask_%d_%d"
        lt_loss = "loss_%d_%d"

        p_lstm_init = {
            "f": ["lstm_init_input_val_f", "lstm_init_input_gate_f",
                    "lstm_init_forget_gate_f", "lstm_init_output_gate_f"],
            "b": ["lstm_init_input_val_b", "lstm_init_input_gate_b",
                    "lstm_init_forget_gate_b", "lstm_init_output_gate_b"]
        }

        p_lstm_infer = {
            "f": ["lstm_input_val_f", "lstm_input_gate_f", "lstm_forget_gate_f",
                    "lstm_output_gate_f"],
            "b": ["lstm_input_val_b", "lstm_input_gate_b", "lstm_forget_gate_b",
                    "lstm_output_gate_b"]
        }

        p_ip_init = ["ip_init_weight", "ip_init_bias"]
        p_ip_infer = ["ip_weight", "ip_bias"]

        p_lstm = p_lstm_init if i_iter == 0 else p_lstm_infer
        p_ip = p_ip_init if i_iter == 0 else p_ip_infer
        p_predict = ["predict_weight", "predict_bias"]

        def do_pass(indices, side, lowers):
            for it in range(len(indices)):
                t = indices[it]
                l_hidden = lt_hidden % (side, i_iter, t)
                l_mem = lt_mem % (side, i_iter, t)
                l_concat1 = lt_concat1 % (side, i_iter, t)
                l_lstm = lt_lstm % (side, i_iter, t)

                if it == 0:
                    l_prev_hidden = l_seed
                    l_prev_mem = l_seed
                else:
                    l_prev_hidden = lt_hidden % (side, i_iter, indices[it-1])
                    l_prev_mem = lt_mem % (side, i_iter, indices[it-1])

                self.net.f(Concat(l_concat1, bottoms=[l_prev_hidden] + lowers[t]))

                #if i_iter == 0:
                #    self.net.f(Concat(l_concat1, bottoms=[l_prev_hidden,
                #            l_features]))
                #else:
                #    l_concat_lower = lt_concat2 % (i_iter-1, t)
                #    self.net.f(Concat(l_concat1, bottoms=[l_prev_hidden,
                #            l_concat_lower]))

                self.net.f(LstmUnit(l_lstm, bottoms=[l_concat1, l_prev_mem],
                        param_names=p_lstm[side], tops=[l_hidden, l_mem],
                        num_cells=N_HIDDEN))

        def do_ip_pass(indices):
            for t in indices:
                l_ip = lt_ip % (i_iter, t)
                l_relu = lt_relu % (i_iter, t)
                if i_iter == 0:
                    self.net.f(InnerProduct(l_ip, N_HIDDEN,
                            bottoms=[l_features], param_names=p_ip))
                else:
                    l_concat_lower = lt_concat2 % (i_iter-1, t)
                    self.net.f(InnerProduct(l_ip, N_HIDDEN,
                            bottoms=[l_concat_lower], param_names=p_ip))
                self.net.f(ReLU(l_relu, bottoms=[l_ip]))

        self.net.f(NumpyData(l_seed, np.zeros((n_batch, N_HIDDEN))))

        #if self.do_forward:
        #    do_pass(list(range(n_waypoints)), "f")
        #if self.do_backward:
        #    do_pass(list(range(n_waypoints-1, -1, -1)), "b")
        #if not (self.do_forward or self.do_backward):
        #    do_ip_pass(list(range(n_waypoints)))

        if i_iter == 0:
            #do_pass(list(range(n_waypoints)), "f", [[l_features]] * n_waypoints)
            do_pass(list(range(20)), "b", [[l_features]] * 20)
        elif i_iter == 1:
            do_pass(list(range(n_waypoints)), "f", [[l_features, lt_hidden % ("b",
                    0, 19)]] * n_waypoints)
        else:
            assert False

        loss = 0
        for t in range(1, n_waypoints):
            l_concat2 = lt_concat2 % (i_iter, t)
            l_predict = lt_predict % (i_iter, t)
            l_loss = lt_loss % (i_iter, t) 
            l_mul_mask = lt_mul_mask % (i_iter, t)
            l_target = ll_targets[t]
            l_mask = ll_masks[t]
            l_prev_predict = lt_predict % (i_iter, t-1)
            l_prev_target = "prev_target_%d_%d" % (i_iter, t)
            n_target = self.net.blobs[l_target].shape[1]

            l_forward = lt_hidden % ("f", i_iter, t)
            l_backward = lt_hidden % ("b", i_iter, t)
            l_relu = lt_relu % (i_iter, t)

            #if self.do_forward and self.do_backward:
            #    self.net.f(Concat(l_concat2, bottoms=[l_forward, l_backward,
            #            l_features]))
            #elif self.do_forward:
            #    self.net.f(Concat(l_concat2, bottoms=[l_forward, l_features]))
            #elif self.do_backward:
            #    self.net.f(Concat(l_concat2, bottoms=[l_backward, l_features]))
            #else:
            #    self.net.f(Concat(l_concat2, bottoms=[l_relu, l_features]))

            if (not self.anytime) and (i_iter < self.n_iters - 1):
                continue

            if self_init:
                if t == 1:
                    data = np.asarray([d.init for d in data])
                else:
                    data = np.round(self.net.blobs[l_prev_predict].data)
                self.net.f(NumpyData(l_prev_target, data))
            else:
                l_prev_target = ll_targets[t-1] 

            self.net.f(Concat(l_concat2, bottoms=[l_forward, l_prev_target,
                    l_features]))
            #l_concat2 = l_forward

            l_ip = "ip_%d" % t
            l_relu = "relu_%d" % t

            self.net.f(InnerProduct(l_ip, N_HIDDEN, bottoms=[l_concat2],
                param_names=p_ip))
            self.net.f(ReLU(l_relu, bottoms=[l_ip]))
            self.net.f(InnerProduct(l_predict, n_target, bottoms=[l_relu],
                    param_names=p_predict))


            self.net.f(Eltwise(l_mul_mask, "PROD", bottoms=[l_predict, l_mask]))
            loss += self.net.f(EuclideanLoss(l_loss, bottoms=[l_mul_mask, l_target]))

        return loss

    def demonstrate(self, data):
        return self.forward(data, train=True)[-1]

    def predict(self, data):
        self.forward(data, train=False)
        out = []
        for i in range(len(data)):
            n_waypoints = len(data[i].demonstration)
            pred = [data[i].init]
            for t in range(1, n_waypoints):
                l_predict = "predict_%d_%d" % (self.n_iters - 1, t)
                pred.append(tuple(self.net.blobs[l_predict].data[i, :]))
            out.append(pred)
        return out
