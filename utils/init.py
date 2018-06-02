from common import *

SEED = 35202 #1510302253  #int(time.time())

def set_results_reproducible():
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def init_torch():
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()   =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device() =', torch.cuda.current_device())