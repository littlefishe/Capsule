class config(object):
    def __init__(self) -> None:
        pass

FLAGS = config()
FLAGS.dataset_type = 'IMAGE100'
FLAGS.model_type = 'VGG16Image'
FLAGS.data_pattern = 0
FLAGS.worker_num = 30 
FLAGS.active_worker_num = 10
FLAGS.round = 1000
FLAGS.labeled_num = 5000
FLAGS.expand = True
FLAGS.pre_round = 50

# global
FLAGS.lr = 0.02
FLAGS.min_lr = 0.005
FLAGS.momentum = 0.9
FLAGS.weight_decay = 5e-4
FLAGS.decay_rate = 0

# server supervised hyperpara
FLAGS.stransform_type = 'strong_twice'
FLAGS.sbatch_size = 128
FLAGS.drop_last = True
FLAGS.ema_decay = 0.99
FLAGS.sup_iters = 50
FLAGS.iter_decay = True

# local
FLAGS.utransform_type = 'fixmatch'
FLAGS.batch_size = 64
FLAGS.local_steps = 50

# cr
FLAGS.uqueue_size = 2048
FLAGS.utemperature = 0.2
FLAGS.cr_threshold = 0.95
FLAGS.emb_dim = 512 * 25
FLAGS.proj_type = 'mlp'
FLAGS.proj_dim = 128


# alg
FLAGS.rho_1 = 0.01
FLAGS.rho_2 = 0.05
FLAGS.milestones = [150, 250, 400, 600, 700]