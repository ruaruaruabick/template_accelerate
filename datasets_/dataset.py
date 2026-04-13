import librosa
import random
import numpy as np
from scipy import stats
import json
import logging
import bisect
from typing import List
import ijson

from torch.utils.data import Dataset,DataLoader,IterableDataset
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class CollateFn:
    def __init__(self, config=None):
        self.config = config
        pass
    
    def __call__(self, batch):
        batch = [b for b in batch if b !=None]

        
        return {
           
        }

class MyData(object):
    def __init__(self, filelist, config=None,):
        ...
    def parsechunk(self, data):
        ...
    
    
class MyDataset(Dataset, MyData):
    def __init__(self,*args, **kwargs):
        MyData.__init__(self,*args, **kwargs)
        Dataset.__init__(self)
        
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            return self.parsechunk(self.data[idx])
        except Exception as e:
            logger.warning(f"Warning in {self.data[idx]}: {e}")
            return None

class MyIterableDataset(IterableDataset,MyData):
    def __init__(self, *args, **kwargs):
        IterableDataset.__init__(self)
        MyData.__init__(self,*args, **kwargs)
        self.gpuid = torch.cuda.current_device()
    
    def merged_chunks(self, filepoints: List):
        for filepoint in filepoints:
            yield from ijson.items(filepoint, 'item')

    def __iter__(self,):
        filepoints = [open(data,encoding='utf-8') for data in self.data]
        ############parse workder set##################
        worker_info = torch.utils.data.get_worker_info()
        self.num_worker = 1 if worker_info is None else int(worker_info.num_workers)
        worker_id = worker_info.id
        readed_num = 0
        ###################iter#################
        for chunk in chunks:
            if readed_num > self.skip_num and readed_num % self.num_worker == worker_id:
                try:
                    result = self.parsechunk(chunk)
                except Exception as e:
                    result = None
                    print(e)
                yield result
            elif readed_num < self.skip_num and readed_num % 10000 == 0:
                self.log(f"skiping,{readed_num},{self.skip_num}")
            readed_num += 1 
        self.log(f"OVER,{readed_num}")