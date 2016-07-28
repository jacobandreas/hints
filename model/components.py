from apollocaffe.layers import *
import numpy as np

def bilstm(prefix, inputs, n_hidden, net):
    ll_forward = list(lstm(prefix + "__forward", inputs, n_hidden, net))
    ll_backward = list(lstm(prefix + "__backward", list(reversed(inputs)), n_hidden, net))
    ll_backward = list(reversed(ll_backward))
    lt_concat = prefix + "__concat_%d"
    assert len(inputs) == len(ll_forward) == len(ll_backward)
    ll_concat = []
    for t in range(len(inputs)):
        l_concat = lt_concat % t
        l_forward = ll_forward[t]
        l_backward = ll_backward[t]
        net.f(Concat(l_concat, bottoms=[l_forward, l_backward]))
        ll_concat.append(l_concat)
    return ll_concat

def lstm(prefix, inputs, n_hidden, net):
    l_seed = "seed_%s" % prefix
    lt_hidden = "hidden_%s_%%d" % prefix
    lt_mem = "mem_%s_%%d" % prefix
    lt_concat = "concat_%s_%%d" % prefix
    lt_lstm = "lstm_%s_%%d" % prefix

    p_lstm = [p % prefix for p in ("lstm_i_val_%s", "lstm_i_gate_%s",
            "lstm_f_gate_%s", "lstm_o_gate_%s")]

    batch_size = net.blobs[inputs[0][0]].shape[0]
    net.f(NumpyData(l_seed, np.zeros((batch_size, n_hidden))))
    l_prev_hidden = l_seed
    l_prev_mem = l_seed
    for t in range(len(inputs)):
        l_hidden = lt_hidden % t
        l_mem = lt_mem % t
        l_concat = lt_concat % t
        l_lstm = lt_lstm % t

        net.f(Concat(l_concat, bottoms=[l_prev_hidden]+inputs[t]))
        net.f(LstmUnit(l_lstm, bottoms=[l_concat, l_prev_mem],
                param_names=p_lstm, tops=[l_hidden, l_mem],
                num_cells=n_hidden))

        l_prev_hidden = l_hidden
        l_prev_mem = l_mem
        yield l_hidden
