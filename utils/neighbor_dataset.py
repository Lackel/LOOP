import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from heapq import nlargest
import openai
import time

class NeighborsDataset(Dataset):
    def __init__(self, args, dataset, indices, query_index, pred, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.query_index = query_index
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pred = pred
        self.count = 0
        self.di = {}
        openai.api_key = args.api_key
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        neighbor_pred = np.take(self.pred, self.indices[index, :])

        res = [neighbor_pred[0]]
        for i in neighbor_pred[1:]:
            if i not in res:
                res.append(i)
                break
        q1 = np.random.choice(self.indices[index, np.where(neighbor_pred==res[0])][0], 1)[0]
        if len(res) == 1 or index not in self.query_index:
            neighbor_index = np.random.choice(self.indices[index], 1)[0]
        else:
            q2 = np.random.choice(self.indices[index, np.where(neighbor_pred==res[1])][0], 1)[0]
            if self.di.get(index, -1) == -1:
                neighbor_index = self.query_llm(index, q1, q2)
                self.di[index] = neighbor_index
            else:
                neighbor_index = self.di[index]

        
        neighbor = self.dataset.__getitem__(neighbor_index)
        output['anchor'] = anchor[:3]
        output['neighbor'] = neighbor[:3]
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor[-1]
        output['index'] = index
        return output
    
    def query_llm(self, q, q1, q2):
        s = self.tokenizer.decode(self.dataset.__getitem__(q)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s1 = self.tokenizer.decode(self.dataset.__getitem__(q1)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s2 = self.tokenizer.decode(self.dataset.__getitem__(q2)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prompt = "Select the customer utterance that better corresponds with the Query in terms of intent. Please respond with 'Choice 1' or 'Choice 2' without explanation. \n Query: " + s + "\n Choice 1: " + s1 + "\n Choice 2: " + s2
        self.count += 1
        # if self.count % 10 == 0:
        #     time.sleep(10)
        
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if 'Choice 1' in completion.choices[0].message['content']:
                return q1
            elif 'Choice 2' in completion.choices[0].message['content']:
                return q2
            else:
                return q1
        except Exception:
            print(Exception)
            return q1


