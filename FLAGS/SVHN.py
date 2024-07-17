class config(object):
    def __init__(self) -> None:
        pass

FLAGS = config()
FLAGS.dataset_type = 'SVHN'
FLAGS.model_type = 'CNN'
FLAGS.data_pattern = 0
FLAGS.worker_num = 30 
FLAGS.active_worker_num = 10
FLAGS.round = 1000
FLAGS.labeled_num = 1000
FLAGS.client_labels = 0
FLAGS.mu = 7
FLAGS.expand = True
FLAGS.pre_round = 0

# global
FLAGS.lr = 0.02
FLAGS.min_lr = 0.005
FLAGS.momentum = 0.9
FLAGS.weight_decay = 5e-4

# server supervised hyperpara
FLAGS.stransform_type = 'twice'
FLAGS.sbz = 64
FLAGS.drop_last = True
FLAGS.ema_decay = 0.99
FLAGS.global_steps = 100

# alg
FLAGS.control = True
FLAGS.alpha = 1.5
FLAGS.beta = 8

# local
FLAGS.utransform_type = 'fixmatch'
FLAGS.ubz = 64
FLAGS.local_steps = 50

# cr
FLAGS.queue_size = 1024
FLAGS.temperature = 0.2
FLAGS.threshold = 0.95
FLAGS.emb_dim = 5*5*64
FLAGS.proj_type = 'mlp'
FLAGS.proj_dim = 128
FLAGS.clear_cache = False
