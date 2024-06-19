import paddle
import models.archs.hdsrnet as hdsrnet


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    scale = opt_net['scale']
    print('*'*100,opt_net)

    # image restoration
    if which_model == 'hdsrnet':
        netG = hdsrnet.DDYCNNSR(scale)


    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG