# numerical libs
import numpy as np
import random

# imaging libs
import cv2
import skimage
import skimage.morphology as skmorph
from scipy import ndimage

# visualizing libs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel.data_parallel import data_parallel

# std libs
import collections
import copy
import numbers
import inspect
import shutil
import csv
import pandas as pd
import pickle
import glob
import sys
import time
import os
import math

# other libs
from tqdm import tqdm