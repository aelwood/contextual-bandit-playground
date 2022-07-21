from __future__ import print_function
import numpy as np
import torch
import torch.nn.init
import torch.nn as nn

gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0

# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(w):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    print (shape, flat_shape)
    q = q.reshape(shape)
    return q.astype(np.float32)

def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    #print('act shape = ', gg['act_dict'].shape)
    return


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #print 'trying to hook to', m, gg['hook_position'], gg['done_counter']
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
            #print ' hooking layer = ', gg['hook_position'], m
        else:
            #print m, 'already done, skipping'
            gg['hook_position'] += 1
    return

def count_conv_fc_layers(m):
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] +=1
    return

def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    return
def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
        else:
            #nn.init.orthogonal(m.weight)
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            #print w_ortho
            #m.weight.data.copy_(torch.from_numpy(w_ortho))
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
    return

def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight'):
                m.weight.data *= float(gg['current_coef'])
                gg['correction_needed'] = False
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data += float(gg['current_bias'])
            return
    return

def LSUVinit(model, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = True,needed_mean = 0., cuda = False, verbose = True):
    xp_batch = torch.tensor([[0.1231, 0.3724],
            [0.0305, 0.3910],
            [-0.0793, 0.3841],
            [1.9342, -0.3238],
            [0.2377, 0.3120],
            [0.1919, 1.9517],
            [0.0148, -0.3920],
            [-0.3675, -1.9264],
            [-1.8444, -0.6666],
            [-1.9291, -0.3529],
            [-0.3398, -0.1958],
            [-1.9255, 0.3723],
            [-0.0540, 0.3885],
            [-0.1723, -1.9536],
            [0.3369, 0.2009],
            [-1.9326, -0.3335],
            [1.1407, 1.5953],
            [-0.2965, 0.2568],
            [-1.8697, -0.5920],
            [1.8840, 0.5448],
            [-1.6843, -1.0047],
            [0.3668, -0.1389],
            [-0.9405, -1.7209],
            [-1.8606, 0.6201],
            [-1.9596, -0.0788],
            [-0.3190, -1.9351],
            [0.0069, 0.3922],
            [0.3041, 0.2477],
            [-1.2727, 1.4921],
            [-1.8894, 0.5258],
            [-0.3708, 0.1278],
            [-0.0384, -0.3903],
            [-0.0540, -0.3885],
            [-0.1324, -0.3692],
            [-0.3757, -0.1127]], dtype=torch.float64)
    yp_batch = torch.tensor([3.9862, 4.2819, 4.5193, 1.0418, 4.4389, 1.3728, 4.7871, 1.0181, 1.5310,
            0.2569, 2.7671, 1.8930, 1.5497, 1.7319, 2.7531, 1.5339, 2.4389, 1.9985,
            1.4253, 1.1395, 1.0283, 4.3547, 0.5513, 1.1992, 1.5296, 1.0110, 4.0623,
            1.7436, 3.0441, 2.6189, 1.3458, 1.2377, 4.6300, 2.8588, 1.0296])



    cuda = False
    gg['total_fc_conv_layers']=0
    gg['done_counter']= 0
    gg['hook_position'] = 0
    gg['hook']  = None
    model.eval();
    if cuda:
        model = model.cuda()
        xp_batch = xp_batch.cuda()
        yp_batch = yp_batch.cuda()
    else:
        model = model.cpu()
        xp_batch = xp_batch.cpu()
        yp_batch = yp_batch.cpu()
    if verbose: print( 'Starting LSUV')
    model.apply(count_conv_fc_layers)
    if verbose: print ('Total layers to process:', gg['total_fc_conv_layers'])
    with torch.no_grad():
        if do_orthonorm:
            model.apply(orthogonal_weights_init)
            if verbose: print ('Orthonorm done')
        if cuda:
            model = model.cuda()
        for layer_idx in range(gg['total_fc_conv_layers']):
            if verbose: print (layer_idx)
            model.apply(add_current_hook)
            out = model(xp_batch, yp_batch)
            current_std = gg['act_dict'].std()
            current_mean = gg['act_dict'].mean()
            if verbose: print ('std at layer ',layer_idx, ' = ', current_std)
            #print  gg['act_dict'].shape
            attempts = 0
            while (np.abs(current_std - needed_std) > std_tol):
                gg['current_coef'] =  needed_std / (current_std  + 1e-8);
                gg['current_bias'] =  needed_mean - current_mean * gg['current_coef'];
                gg['correction_needed'] = True
                model.apply(apply_weights_correction)
                if cuda:
                    model = model.cuda()
                out = model(xp_batch, yp_batch)
                current_std = gg['act_dict'].std()
                current_mean = gg['act_dict'].mean()
                if verbose: print ('std at layer ',layer_idx, ' = ', current_std, 'mean = ', current_mean)
                attempts+=1
                if attempts > max_attempts:
                    if verbose: print ('Cannot converge in ', max_attempts, 'iterations')
                    break
            if gg['hook'] is not None:
                gg['hook'].remove()
            gg['done_counter']+=1
            gg['counter_to_apply_correction'] = 0
            gg['hook_position'] = 0
            gg['hook']  = None
            if verbose: print ('finish at layer',layer_idx )
        if verbose: print ('LSUV init done!')
        if not cuda:
            model = model.cpu()
    return model