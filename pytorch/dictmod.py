import torch


def remove_module_prefix(state_dict):
    keys = []
    new_dict = dict(state_dict)
    for key in new_dict:
            keys.append(key)
    
    for key in keys:
        if key[:7] == 'module.':
            new_dict[key[7:]] = new_dict[key]
            del new_dict[key]
    
    return new_dict


def rename_final_conv(state_dict):
    state_dict['out_tr.final_conv.0.weight'] = state_dict['out_tr.final_conv.weight']
    state_dict['out_tr.final_conv.0.bias'] = state_dict['out_tr.final_conv.bias']
    del state_dict['out_tr.final_conv.weight']
    del state_dict['out_tr.final_conv.bias']
    
    return state_dict

def modify_statedict(state_dict, in_name, new_in_ch, out_name, new_out_ch, remove_key_prefix=True, rename_final=False):
    if remove_key_prefix:
        new_dict = remove_module_prefix(state_dict)
    else:
        new_dict = dict(state_dict)
    
    if rename_final:
        new_dict = rename_final_conv(new_dict)
    
    assert((in_name+'.weight' in state_dict) or (in_name+'.weight' in new_dict))
    assert((out_name+'.weight' in state_dict) or (out_name+'.weight' in new_dict))
    assert((out_name+'.bias' in state_dict) or (out_name+'.bias' in new_dict))

    #print(new_dict[in_name+'.weight'].shape)
    #print(new_dict[out_name+'.weight'].shape)
    #print(new_dict[out_name+'.bias'].shape)
    
    new_dict[in_name+'.weight'] = new_dict[in_name+'.weight'].repeat(1, new_in_ch, 1, 1, 1)
    new_dict[out_name+'.weight'] = new_dict[out_name+'.weight'].repeat(new_out_ch, 1, 1, 1, 1)
    new_dict[out_name+'.bias'] = new_dict[out_name+'.bias'].repeat(new_out_ch)
    
    #print(state_dict['module.'+in_name+'.weight'].shape, new_dict[in_name+'.weight'].shape)
    #print(state_dict['module.'+out_name+'.weight'].shape, new_dict[out_name+'.weight'].shape)
    #print(state_dict['module.'+out_name+'.bias'].shape, new_dict[out_name+'.bias'].shape)

    return new_dict

if __name__ == '__main__':
    import sys
    import os
    new_state_dict = modify_statedict(torch.load(sys.argv[1],map_location='cpu')['state_dict'], 'down_tr64.ops.0.conv1', 5, 'out_tr.final_conv', 7)
    torch.save(new_state_dict, os.path.splitext(sys.argv[1])[0]+'_'+str(5)+'in_'+str(7)+'out.pth')