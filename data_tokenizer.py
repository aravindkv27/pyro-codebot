import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import *
from torchtext import data

import numpy as np
import pandas as pd

import random
import math
import time
from tokenize import tokenize, untokenize
import io
import keyword
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchtext.data import BucketIterator

# Question starts with '#'

## Using a custom tokenizer to tokenize python code


f = open("data.txt", encoding='utf-8')
file_lines = f.readlines()



dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    dp['question'] = line[1:]
  else:
    dp["solution"].append(line)

def run_on_gpu():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

def tokenize_python(python_code_str):

    tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))

    tokenized_op = [(token.type, token.string) for token in tokens]
    
    return tokenized_op


def augment_tokenize_pycode(python_code_str, mask_factor=0.3):

    var_dict = {}
    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip', 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)
    var_counter = 1
    
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_op = []
    
    for i in range(0, len(python_tokens)):
        if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:
            
            if i > 0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']:
                skip_list.append(python_tokens[i].string)
                tokenized_op.append((python_tokens[i].type, python_tokens[i].string))

            elif python_tokens[i].string in var_dict:
                tokenized_op.append((python_tokens[i].type, var_dict[python_tokens[i].string]))

            elif random.uniform(0, 1) > 1 - mask_factor:
                var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
                var_counter += 1
                tokenized_op.append((python_tokens[i].type, var_dict[python_tokens[i].string]))

            else:
                skip_list.append(python_tokens[i].string)
                tokenized_op.append((python_tokens[i].type, python_tokens[i].string))

        else:
            tokenized_op.append((python_tokens[i].type, python_tokens[i].string))
    
    return tokenized_op


def train_validation():

  python_problems_df = pd.DataFrame(dps)

  # Splitting data into 80% train and 15% validation
  split_data = np.random.rand(len(python_problems_df)) < 0.80

  train_df = python_problems_df[split_data]
  val_df = python_problems_df[~split_data]

  return train_df, val_df

