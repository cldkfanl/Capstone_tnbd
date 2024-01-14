# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import inspect
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from typing import Optional
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils import TryExcept
from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))

# Settings
DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return bool(re.search('[\u4e00-\u9fff]', str(s)))


def is_colab():
    # Is environment a Google Colab instance?
    return 'COLAB_GPU' in os.environ


def is_kaggle():
    # Is environment a Kaggle Notebook?
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_docker() -> bool:
    """Check if the process runs inside a docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:  # check if docker is in control groups
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    if is_kaggle() or is_colab():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("yolov5")  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class Timeout(contextlib.ContextDecorator):
    # YOLOv5 Timeout class. Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = True  # for faster training
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_age(path=__file__):
    # Return days since last file update
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo='ultralytics/yolov5', branch='master'):
    # YOLOv5 status check, recommend 'git pull' if code is out of date
    url = f'https://github.com/{repo}'
    msg = f', for updates see {url}'
    s = colorstr('github: ')  # string
    assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
    assert check_online(), s + 'skipping check (offline)' + msg

    splits = re.split(pattern=r'\s', string=check_output('git remote -v', shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = 'ultralytics'
        check_output(f'git remote add {remote} {url}', shell=True)
    check_output(f'git fetch {remote}', shell=True, timeout=5)  # git fetch
    local_branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {local_branch}..{remote}/{branch} --count', shell=True))  # commits behind
    if n > 0:
        pull = 'git pull' if remote == 'origin' else f'git pull {remote} {branch}'
        s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `{pull}` or `git clone {url}` to update."
    else:
        s += f'up to date with {url} ✅'
    LOGGER.info(s)


def check_python(minimum='3.7.0'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING: ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


@TryExcept()
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=''):
    # Check installed dependencies meet YOLOv5 requirements (pass *.txt file or list of packages or single package str)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            s += f'"{r}" '
            n += 1

    if s and install and AUTOINSTALL:  # check environment variable
        LOGGER.info(f"{prefix} YOLOv5 requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            assert check_online(), "AutoUpdate skipped (offline)"
            LOGGER.info(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} {e}')


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        LOGGER.warning(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    elif file.startswith('clearml://'):  # ClearML Dataset ID
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT, progress=False):
    # Download font to CONFIG_DIR if necessary
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    # Download, check and/or unzip dataset if not found locally

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir=f'{DATASETS_DIR}/{Path(data).stem}', unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Checks
    for k in 'train', 'val', 'names':
        assert k in data, f"data.yaml '{k}:' field missing ❌"
    if isinstance(data['names'], (list, tuple)):  # old array format
        data['names'] = dict(enumerate(data['names']))  # convert to dict
    data['nc'] = len(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info('\nDataset not found ⚠️, missing paths %s' % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception('Dataset not found ❌')
            t = time.time()
            root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
            if s.startswith('http') and s.endswith('.zip'):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f'Downloading {s} to {f}...')
                torch.hub.download_url_to_file(s, f)
                Path(root).mkdir(parents=True, exist_ok=True)  # create root
                ZipFile(f).extractall(path=root)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', root)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf', progress=True)  # download fonts
    return data  # dictionary


def check_amp(model):
    # Check PyTorch Automatic Mixed Precision (AMP) functionality. Return True on correct operation
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        # All close FP32 vs AMP results
        m = AutoShape(model, verbose=False)  # model
        a = m(im).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(im).xywhn[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance

    prefix = colorstr('AMP: ')
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices
    f = ROOT / 'data' / 'images' / 'bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend('yolov5n.pt', device), im)
        LOGGER.info(f'{prefix}checks passed ✅')
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(f'{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}')
        return False


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        success = True
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            LOGGER.info(f'Downloading {url} to {f}...')
            for i in range(retry + 1):
                if curl:
                    s = 'sS' if threads > 1 else ''  # silent
                    r = os.system(
                        f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -')  # curl download with retry, continue
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f'Download failure, retrying {i + 1}/{retry} {url}...')
                else:
                    LOGGER.warning(f'Failed to download {url}...')

        if unzip and success and f.suffix in ('.zip', '.tar', '.gz'):
            LOGGER.info(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.tar':
                os.system(f'tar xf {f} --directory {f.parent}')  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using w�y#htef$ke`n)
$( !`"  `0   u0tate BO�e� aq boYes)m,)�= w%ighwc(y,n)�� boxes(n)
 ��     !   kme8= bni_iK}(roh!r[)]<``oxe�,0> hgu^t�p�s@ #6i�5	iieriX�   h 0  $h �wgishtr =!i�u`* score�[No.u_ "!abo� ueic`|c     �`   ( x�a, .]5 torsh.em(�dhghps,`zY:, "t])nvloavh)!/!wem�jts.que�1, immp$km=Tuw)(8�$}ergud)ropes
  $�   $   iv vedult1nt�
 ` 0 !     $ (  i - a[iou/wU�(�) $1] # ReqUipe`�ed�ndqnc�

A" �$ "m�p8up[xm] = x[A�
  a $  0�n *dm-%/zk�g() � �) 6 zkme_li-it: ! !�" ( ) �LRFGDZ5ar�ing>e'WASnG:�N	_ w�Me limit@{p)mG_limmt:&3vr"isaaded'8 0  (!  #$�"breaj �#(`ime!dmmiT gxgeedee    fEpuvl wutpUt


$ef`ctrop_op|iiiz�r,f='deSt�p6,1s}''�>  # fanmdQtyhs
gcne2en )mport *9 swripopt)m�x�r((J  ( # st:iP optim{zd�$fr/l %v'`|o!jk�%lmzc zainioa, kxtilnally%s`rm`as#3'
   &x!0t�rci.$oa$(&m lah�`mcaTiKj=>or�h.devibeh�cpu')I
, � if`RgEr8&e}`g9:*  �  $ �y['m�deh7]  xS%eep&Y` 3 r!p|aCe!moD�aw�dh�em��!` 6or k0i~ 'optim)jEr', -bus~f��fess>, 'w�fdbWm$'�b'ema'.,'wpee�%s'; �� �ei3
h !   , |[o]`= Kone
   8�Z/ep[C`'\ �$-5
 !  x[gmoL�l]>hal�()"#!to`Bн2
2  (n�r`p (' xk'mod�|'].pazamEtE2s():
     "$ p.besemres[gta$ = a<sa
 �) to��H.cavu(xh0c or$f�
 �$eb 9 +s.qathjg`ps�ze(r er �)8?$1E6$ # FilQiZe
   (LGGFEr.i-g�(n&Op4im)Z%r$s4ri8peD)f�om Uf},{f'!savud As({s4' io�S Gl#e ''} {ic:.1d}MB"i

du&�pvint}mtvavi/n(rE3}n4q `yp,8saTe�eyr$�fqck�u| q2efhx=cojorqvv('evolVe8 ')i�
   (m6olre^CsV ?*s�we_di�/�G%vo,ve.c�F�
    evolveaa=n < wawdd)z,/"'hYpWevml&e.iamh'* 0$keyc�= )'M�d:Xcs/pwdc�syo>&. ;m�dris/rgc!dl'1mavsics/aQP_1&5', �Meprmcc/-�@[�5:�.9%'l 'val'boX_Lo�s�,
    ! $``  'val-ck_lo{c,0'vaD-;lw_lo��'�(+$ttale(hy�.k!}s8�+4 #0Yrg�ults +0hy0S]
� ` ceY2 - tuple(x�wvZap(- tor y in ogys(   `val{a=02us]$ds`� tu�le(iyp.vq�qqs()(�    n 5 len(�eyw)
  $ #$Dow�oo�dH(Np|9oa|)
"  $yf buc;et:
( `�c (uRl�} f'ec//rujku�/tvolTu/cst&
  �  ""(ab$gsuTin_cevsibG�url!$!(evOl6e]gsw.3tat(	Ns4Mq(ze"av evolue[gyv.eXAstS*) elr%(0+:
 8 �#  (`  "oQ.Sqs0em(f%gsut,l #t s5rly(;ca�e_di�}')($#!dlNload �Vodvu.csv(if�largmr |�Ef�local

 "  "0Log"w� ewL�vg.�st
" `!w"= ''�if mvomv%_ccv.dxiC�c()0exsm (,/%2ps-' * nd5�keYsi�rst�ip�7,% + 'TF) "#$cdl$hea�%BJ !"�w�th�mpg�evo|ue]ss6,$7!#) ac�V:  ` " j f.w�iu%(S$3 )'%:0.5g$g�j`l"% va�q**�s�rap�/,')!+ '\n-

 `  0Save yaml
0 � with opdn(enoLr�_ka-l�1o�')"as $; (% $ !�$`dc0= rdnSea_s3�(golwm_bsv/
a"0�  ( datc0= da|anRdnam�(clUmns=la�`d3�x: x.strax())( '(sTrip cei�
 8   �  i =(n0.`pgmap(fivle�s(lq�!.6ahues[;,!:4])) `2   �!h  gcmera|kEnw0?`m��8dava) 00 ` ` fwVmve,� �OLOv5 Di�uwpprimgue"$evelUu�}j Vmsul�c\n.�+ f'# @os4�genefcvion: {k}\n'8�
" 0  (    ((  ! D'"4Last(generatign:0{ge�drad}onq - 0}\F' k+#(' + �,<'.Joil(f'Sx�c�skp�9:> s?'`f�r x ij$ieys:7_m ;
 ` h`    (0 !$  7\n5 2g# ' � #< &.noin(v#s8Z>2�,�G�/ fnr 8 in `1�a/v��ugs[i, :�]) +�G\n\n'�
" ` �   ya}i.S!fe_euMp(�apalgc[�S7M.To_digu)	�f<`coRt��eys=F!lsd)
 �  # prile0to(sb`gq>J"  *LKGGER.iv$k(pre^ip + '';gefe�atig/{] geneRatimn� vmnis@ed�"ctr�endlbcSultLn' + qrefiz0+�      0�" �$    - '.j~an$fG{x.rtrir,)8?20sm' fo�dx n0jexsI + '|n' +!xrefip + '$ '.jOin(f'{x:20.5w}gJ   (��("   ba�  ` �     ,$�       "  $�$�   2!0  �" !�  � !$` "    '�!  `   �  $�` `0 f�r h`in fAm�k + &|n�n�9
J    �f bucket:
  �     o�.cistem,fw{�ti.0cq({mrO�veOcqv} �evglvaOy�-l] g1~/�bubkuw}'	  ' uplmab
Jdef`aq�my_cl`�[idia2)x,mkde,, kog,(im�-*
  00# A`pl�+a$sEcozd$�tafe classif�er to"YO|O �u1pu4{
r` !# eyqm0lm mode� =!toRcIr)�ioomodel�.__dmkp__�'effibientN%4]b0�],pxatwainef5Tr�e-.d�(tMviau).uv`l))   ham0 =1[i-p_ YV`iSinst��3h,y{8< np..darzq) el3e(im0.    Fov #, d h. enuo%r�Em(p): �� pe� im!gd
  �" `4 In d i! jkt Oofm�alt l5n(f!8
 `!  0 0�`  d � t.cl�Ne()
` 0$ �   !  # ZdS(!`e clD pAd sUdlwts � %  `   ` b = xy8�rXywh)F[
,`:0])  ! `oxew  `�      0`[:$ 2:M$}b[:.02]./bx(1-Z��uf{!uEe~%(11  # rectanglq"rn {quarG
`8!%  !     f[:, 3>] = b[:,!<2]"*`0*3 + �0 �# �q`
(   !"$0    d[;,0:4y } xy�`"x�xi(�).dn~g(9�
2"2!a a   "3 ZescAle boxa#(fpom&imgOsize�tg �|00saze` !   "0!   q`�}e_sokrds(y�g&shype[2;\ d[:( ��], im9_)]*chap�i
*8 `     `  # Khac���
  0 $(    0 �redClw1 = `[;, ==�loou�
 `   &( $  �Im3 < [M
% 3�     (!(e_r a"Al t:
       0  �    c�uou�(=0aI0[Y][hnt8aI1\-2qft(i;3Y),%ilt<a[0Y	;an|iaOs])�
�"(�   `!  00 ( il(5 cvr.res�ze(cu`ouu$!�r�, 224))� # BCB
     $   � �b& $ii�= iaQ:, :l ::o1I.tsAnyp?se(0, 0, 3) "#�BB`�o"RGf- Uo 3x41>x416
�  �     8    ""�m =�npcscontio�oewabr �(i�,�dt9pe=np+flka�2!( # uYnt("dg$�moCt#22$  .!  � !  d)�im O= 254  #�< -!"55 vo 0.$ - 1.0�(  " *  �0    ( �ms,ap0e�dlie	
(!  h ! (   v2�`�cLs�  modulhtovch.TefSory�s).eo8dn@!wibe)).a2cmax()  # �D`sskfigs qra�a#t)�n� 8  (  �  " xSaU" x[IYNqr�dc�s5 ==`pbmt_cm{2 8" pa�`in$matclmnc!class $�te&$i/n

  0�r%tur�$X
duf!mncreme,tOp!tj(�ath("exi[tmk=Walse,0wup=''(f-kdi2=Vq,�e)    c I~crmdnp vyl%(or �krectoR� path��d� �un3oexP"�)> b5nS/%x�{tep}2� relS/�zp{seq}�&0/&. atc*��   dCth =`Xath(`atl�  # o7-!wngs4(a0! (ir Pqtze8ys�s()ba,� n~t$uxhrD_ojz
  !    `p��h, c�ffIx  xA�H>�i�H��udnix(&&)$ PA�hsuf�ix)$if�pathn)s_�#n]() mfce (pcth$�1) H A   1  ' Methol 1
   (`h �$/r n in!rA~k$(;,9199h:�"        p =jf&{�aT�{sep}kny{+u�fix}&  # xngzeeenT path
   $�     ` ig .k�$nc.p-pj.epi7vs p):  :
   @  (       `jbreik
( a!` � ratH$=0Xath(p) 
`       # MeThod 2"*dErreccp�|)
%  0   + dirs = g�o�.glof:�6kpAth=[s%p}*b)  '!simalqr(tqdhs
0$�  ��  mat#hes 0[r�.q%arc�mrf {patH/s`uo={sep=�\D�" d((v�r � in d�rs] $ ` � (�`i"= [mlt(m.gr^u0s(){0=) dor m i^ matches"if`M] �`imdicEs
 !  !  0k o$= mazi!$�`1 if i!g�se82 %#!inqx%mej4$cumfwz
`#  0 $ �pe�� % �etH(n"{raxl}{seq|{n}sstffix?`!  ��kncreoent&`At�

�`  if mkbIr8
�$      SiTh/mk�)rpyrents-TruE,�a��svOok5Tv�%)��#"oa��"diresuN�9

 �  rgt5rn�peth�
3 MsejC^"C|ynese/f2ieFd�x"fwocpiOks 5�/%-m--m=-9---,--<---)'=--/)--<+-,�-�e---l%)--�--/----�-%----i---,-9,%--,/-*km3hou_<(o2*imr gs! #0copY�|O �v/ie b%+}2wakl �rrous
J
�ef`amread(patl8f|p/s=cv:.YL�e�D_C�OR):  $ !rePuJn(sR2?Am�ecodmo�&fR=ld)le*�1th, j0nuMn48),`fmawr!

*ddf imwzIte(Qath,`k�):
 � (vry2(` �    kv2.yNefcou% Rauh(pyth).suf~iH,`Im)۱}>tofILe�uqtH�
  * ��0 rexu2n!Trtm�" ` mxSePt 8cepUio�:$`0 �"  zevuzn FonsE

d%d�a�siKwhpathL$ie):` ( kmsbow_8pdth.$nsode'|�ibodE_fscape')NdEcodE(), im/
�v".��sead& cv2*i�urit�l`#v2.imsHOw$= imread,(i�sRive, mmchnw 2"0retefi�e

# Vcsi@flaq!-=)--=��--,---)----=�-/--,---	-------M---------�------=---�%--�,%!-/--=-/-=->-%)---------)-��-----�----)oCOHS 9�0 if i{^d+ck�r)(�mse 7Hepi|.'%|[�armh�am_size()fconu]jc 03(�erm�nAl w�ndo� siz� �kr tqdm