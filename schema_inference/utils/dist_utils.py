from multiprocessing import Queue


class DistLaunchArgs:
    def __init__(
        self,
        ngpus_per_node: int,
        world_size: int,
        distributed: bool,
        multiprocessing: bool,
        rank: int,
        seed: int,
        backend: str,
        master_url: str,
        preserve_gpu: bool = False,
        preserve_percent: float = 0.99,
        use_amp: bool = False,
        debug: bool = False,
    ):
        self.ngpus_per_node = ngpus_per_node
        self.world_size = world_size
        self.distributed = distributed
        self.multiprocessing = multiprocessing
        self.rank = rank
        self.seed = seed
        self.backend = backend
        self.master_url = master_url
        self.preserve_gpu = preserve_gpu
        self.preserve_percent = preserve_percent
        self.use_amp = use_amp
        self.debug = debug


class LogArgs:
    def __init__(
        self,
        logger_queue: Queue,
        logdir: str,
        filename: str,
        ckpt_path: str
    ):
        self.logger_queue = logger_queue
        self.logdir = logdir
        self.filename = filename
        self.ckpt_path = ckpt_path
