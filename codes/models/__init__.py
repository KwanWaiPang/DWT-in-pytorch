def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
        # from .SR_model_first import SRModel as M
    elif model == 'sr_sub':
        from .SR_subnet_model import SRModel as M
    elif model == 'sr_noise_blur':
        from .SR_noise_blur_model import SRModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
