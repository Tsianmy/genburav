import torch

def GaussSplatOptimizer(
    base,
    model,
    position_lr=0.00016,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    *args,
    **kwargs
):
    def replace_tensors(self, tensors_dict):
        for group in self.param_groups:
            if not group['name'] in tensors_dict: continue
            shape = tensors_dict[group['name']]
            stored_state = self.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"].new_zeros(shape)
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"].new_zeros(shape)

    def cat_tensors(self, tensors_dict):
        for group in self.param_groups:
            if not group['name'] in tensors_dict: continue
            assert len(group["params"]) == 1
            shape = tensors_dict[group["name"]]
            stored_state = self.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], stored_state["exp_avg"].new_zeros(shape)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], stored_state["exp_avg_sq"].new_zeros(shape)), dim=0)

    def prune_tensors(self, names_list, mask):
        for group in self.param_groups:
            if not group['name'] in names_list: continue
            stored_state = self.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask].contiguous()
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask].contiguous()
    
    def sync_model_params(self, params, operations):
        for op in operations:
            if op['type'] == 'replace':
                self.replace_tensors(op['shape'])
            elif op['type'] == 'add':
                self.cat_tensors(op['shape'])
            elif op['type'] == 'prune':
                self.prune_tensors(op['params'], op['mask'])
        for group in self.param_groups:
            if not group['name'] in params: continue
            param = params[group['name']]
            stored_state = self.state.get(group['params'][0], None)
            if stored_state is not None:
                del self.state[group['params'][0]]
                self.state[param] = stored_state
            group['params'] = [param]
    
    _base_cls = getattr(torch.optim, base)
    _base_cls.replace_tensors = replace_tensors
    _base_cls.cat_tensors = cat_tensors
    _base_cls.prune_tensors = prune_tensors
    _base_cls.sync_model_params = sync_model_params

    l = [
        {'params': [model._xyz], 'lr': position_lr, 'lr_scale': model.cameras_extent, "name": "xyz"},
        {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
        {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
        {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
        {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
        {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
    ]
    optimizer_instance: torch.optim.Optimizer = _base_cls(l, *args, **kwargs)
    optimizer_instance.register_step_pre_hook(model.optimizer_step_pre_hook)
    return optimizer_instance