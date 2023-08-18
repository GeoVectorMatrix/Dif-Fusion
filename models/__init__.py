import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_fusion_model(opt):
    from .Fusion_model import DFFM as M
    m = M(opt)
    logger.info('Fusion Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
