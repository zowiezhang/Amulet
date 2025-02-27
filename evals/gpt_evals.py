import os
import json
from copy import copy
import multiprocessing
from multiprocessing import Manager, Pool

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import args
from utils.gpt_eval import gpt_winner_evaluator


class MatrixBuilder():

    def __init__(self, args):
        self.args = args
        self.dir_path = os.getcwd()
        self.dimen = args.pref_name
        self.model_name = args.model_name
        self.dataset_name = args.eval_data
    
    def save_data(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path)

    def save_json(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w+') as f:
            json.dump(data, f)
    
    def load_df_data(self, file_path):
        data = pd.read_csv(file_path)

        return data

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def get_gpt_eval(self, player_names_list):

        players_list = [self.load_data(os.path.join(self.dir_path, \
                        f"responses/{self.dimen}/{self.model_name}/{self.dataset_name}/{player_name}.json")) \
                        for player_name in player_names_list]

        mtx = self.get_matrix(players_list[-1], player_names_list, players_list)
        
        print(f"Results of {self.dimen}-{self.model_name}-{self.dataset_name} is:")
        print(mtx)

    def process_data_point_worker(self, args):
        mtx, data_idx, data_point, player_names_list, players_list = args
        for baseline_idx in range(len(player_names_list)):
            # amulet data should be the last one in players_list
            win_tag = gpt_winner_evaluator(
                        data_point['question'],
                        f'Your answer should be {self.dimen} as much as possible.',
                        players_list[-1][data_idx]['response'],
                        players_list[baseline_idx][data_idx]['response']
                    )
            if win_tag == 1:
                mtx[baseline_idx][0] += 1
            elif win_tag == -1:
                mtx[baseline_idx][2] += 1
            else:
                mtx[baseline_idx][1] += 1

        if (data_idx + 1) % 10 == 0:
            print(f'===================Finished {data_idx + 1} data points!==========================')
            print(np.array(copy(mtx)))
        return None

    def get_matrix(self, dataset, player_names_list, players_list):
        manager = Manager()

        mtx = manager.list([manager.list([0] * 3) for _ in range(len(player_names_list))])

        with Pool(processes = multiprocessing.cpu_count()) as pool:
            
            params = [
                (mtx, data_idx, data_point, player_names_list, players_list)
                for data_idx, data_point in enumerate(dataset)
            ]

            # Use imap to process each iteration in parallel
            results = pool.imap(self.process_data_point_worker, tqdm(params))
            r = [_ for _ in results]

        pool.close()
        pool.join()  
        
        rsts = pd.DataFrame(np.array(mtx), index = player_names_list, columns = ['win', 'tie', 'lose'])

        self.save_data(rsts, os.path.join(self.dir_path, f'results/gpt_win_rate/{self.dimen}/{self.model_name}/{self.dataset_name}/gpt_win_rate_mtx.csv'))

        return np.array(mtx) 


if __name__ == '__main__':
    evaluator = MatrixBuilder(args)
    player_names_list = ['base', 'pref', 'beam', 'la', 'amulet']
    evaluator.get_gpt_eval(players_list[0], player_names_list)
    







