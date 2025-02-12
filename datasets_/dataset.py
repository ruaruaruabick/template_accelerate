import librosa
import random
import numpy as np
from scipy import stats
import json
import logging
import bisect

from torch.utils.data import Dataset,DataLoader
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
    
class MyDataset(Dataset):
    def __init__(self,filelist, config=None, use_cache=False):
        self.data = filelist 
        self.config = config
        self.use_cache = use_cache

        self.jsoncache ={} 
        
        pass

    def __len__(self):
        return len(self.data)
    
    
    
    
    def __getitem__(self, idx):
        try:
            return self.__tgetitem__(idx)
        except Exception as e:
            logger.warning(f"Warning in {self.data[idx]}: {e}")
            return None
    def __tgetitem__(self, idx):
        data = self.data[idx]
        
        
        
        return {

        }


    
##############TEST##############
if __name__ == '__main__':
    from dataset import MyDataset
    import yaml
    from tqdm import tqdm
    import json
    from multiprocessing import Pool, Process
        
    def load_json(file_path,use_split=False, index=None):
        if use_split:
            assert index is not None
            file_path = file_path.replace('.json', f'_{index}.json')
            if index == 0:
                print('###################Split Training Data###################')
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data 
    #load config
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataconfig = config['dataset']
    #load data
    train_list, valid_list, test_list = load_json(dataconfig['train_file']), load_json(dataconfig['valid_file']), load_json(dataconfig['test_file'])
    for list in [train_list]:
        training_dataset = MyDataset(list,dataconfig)
        # #iter dataset
        # maxlen = training_dataset.__len__()
        
        # num_process = 64
        # patch = maxlen // num_process
        # patch += 1 if maxlen % num_process != 0 else 0
        
        
        # def check(idx):
        #     num = 0
        #     Nonenum = 0
        #     for i in tqdm(range(patch*idx,min(patch*(idx+1),maxlen)),disable= (idx!= 0)):
        #         if training_dataset.__getitem__(i) != None:
        #             num += 1
        #         else:
        #             Nonenum += 1
        #     print(num, Nonenum, patch)        
            
        # process_list = []
        # for i in range(num_process):
        #     p = Process(target=check, args=(i,))
        #     p.start()
        #     process_list.append(p)
        # for p in process_list:
        #     p.join()
        # pass
    
        # iter dataloader
        train_dataloader = DataLoader(training_dataset, batch_size=64, num_workers=64, shuffle=True, collate_fn=CollateFn(dataconfig['collate']))#
        for i in tqdm(iter(train_dataloader)):
            # break
            pass
        
