import json
import os
import shutil

from meta_learning import lstm_seq2seq
import learn2learn as l2l
import numpy as np
import ot
import torch
import torch.nn.functional as F
from pandas import DataFrame
import pandas as pd
from meta_learning.MAML_by_learn2learn import load_the_task_tree_DFS


similarity_matrix_distribution = []
similarity_matrix_spatial = []
similarity_matrix_learning_path = []
training_data = []


def traverse_old(node, file_name):
    if os.path.exists(node.parameter_path + "data\\"):
        file_list = os.listdir(node.parameter_path + "data\\")
        if file_name in file_list:
            return node.parameter_path
        else:
            return None
    else:
        if node.children:
            for child in node.children:
                result = traverse_old(child, file_name)
                if result is not None:
                    return result



class LearningTask:
    def __init__(self, ID, meta_learner, parameters, flag="train", path="", flag_l_p=True, dataset="Hannover",
                 flag_s_f=True):

        self.ID = ID
        self.learner = meta_learner
        self.args = parameters
        self.dataset_name = dataset
        self.flag = flag
        self.data_set = self.read_the_learning_task_dataset(self.ID)
        self.k_step_learning_path = read_learning_path(self, flag_l_p, 30)
        self.spatial_feature = spatial_feature_construction(self, path, flag_s_f)
        self.cluster = None
        self.utility = 0  # only when the cluster is not None self.utility is useful

    def read_the_learning_task_dataset(self, user_id):

        data_list = []

        if self.dataset_name == "Porto_Grid":
            if self.flag == "train":
                file_path = "train\\task.txt"
            else:
                file_path = "test\\task.txt"
            if os.path.exists(file_path):
                with open(file_path) as file:
                    for line in file:
                        strings = line.replace("\n", "").split(",")
                        if len(strings) == 1:
                            continue
                        else:
                            data_list.append([float(strings[1]), float(strings[2])])
                return data_list
        return None


def read_learning_path(task, flag_l_p, sampling):
    """
    read the learning path from file: learning_path.csv
    @return:
    """
    if flag_l_p:
        if task.dataset_name == "Hannover":
            file_list = os.listdir("D:\\数据集\\Hannover_Trajectory\\learning_path\\" + task.args.model_name + "_" + str(sampling) + "\\")
            if str(task.ID) in file_list:
                path = "D:\\数据集\\Hannover_Trajectory\\learning_path\\" + task.args.model_name + "_" + str(sampling) + "\\" + str(task.ID) + "\\"
            else:
                path = "D:\\数据集\\Hannover_Trajectory\\learning_path\\" + task.args.model_name + "\\" + str(task.ID) + "\\"
            # path = "D:\\数据集\\Hannover_Trajectory\\learning_path\\" + task.args.model_name + "_" + str(sampling) + "\\" + str(task.ID) + "\\"
            data_np = np.loadtxt(path + "learning_path.csv", delimiter=',')
            learning_path = torch.from_numpy(data_np)
            return learning_path
        elif task.dataset_name == "Porto":
            if argparser_return().facter == "distribution":
                return None
        elif task.dataset_name == "Porto_Grid":
            if task.flag == "train":
                path = "D:\\数据集\\Porto_Taxi\\learning_path\\train\\" + str(task.ID) + "\\"
            else:
                path = "D:\\数据集\\Porto_Taxi\\learning_path\\test\\" + str(task.ID) + "\\"
            # path = "D:\\数据集\\Porto_Taxi\\learning_path_pami\\train\\" + str(task.ID) + "\\"
            # path = "D:\\数据集\\Porto_Taxi\\learning_path_random\\train\\" + str(task.ID) + "\\"
            if os.path.exists(path):
                data_np = np.loadtxt(path + "learning_path.csv", delimiter=',')
                learning_path = torch.from_numpy(data_np)
                return learning_path
            # if argparser_return().facter == "distribution":
            #     return None
    else:
        return None


def spatial_feature_construction(task, path, flag_s_f):
    if flag_s_f:
        if task.dataset_name == "Hannover":
            # spatial_feature = []  # POI sequence of user
            if task.flag == "train":
                # data_path = "D:\\李慧岭\\.code\\Time-Series-Library-main\\dataset\\Geolife_Trajectories_1.3\\meta\\train\\"
                data_path = path + "train\\"
            else:
                # data_path = "D:\\李慧岭\\.code\\Time-Series-Library-main\\dataset\\Geolife_Trajectories_1.3\\meta\\test\\"
                data_path = path + "test\\"

            data_path = "D:\\数据集\\Hannover_Trajectory\\poi\\sampling_30\\"
            df_POI = pd.read_csv(data_path + str(task.ID) + ".csv")
            cols = list(df_POI.columns)
            cols.remove("Unnamed: 0")
            df_POI = df_POI[cols]
            return df_POI
        elif task.dataset_name == "Porto":
            if argparser_return().facter == "distribution":
                return None
        elif task.dataset_name == "Porto_Grid":
            if task.flag == "train":
                data_path = "D:\\数据集\\Porto_Taxi\\POI\\train\\"
            else:
                data_path = "D:\\数据集\\Porto_Taxi\\POI\\test\\"
            if os.path.exists(data_path + str(task.ID) + ".csv"):
                df_POI = pd.read_csv(data_path + str(task.ID) + ".csv")
                cols = list(df_POI.columns)
                cols.remove("Unnamed: 0")
                df_POI = df_POI[cols]
                return df_POI
        return None
            # if argparser_return().facter == "distribution":
            #     return None


def spatial_similarity_calculator(task1: DataFrame, task2: DataFrame):
    """
    calculate the spatial similarity of two learning tasks
    @param task1: the spatial feature(POI sequence) of learning task 1 [(date, latitude, longitude, type)]
    @param task2: ~~~ task 2
    @return: the spatial feature similarity
    """

    def kernel_function(tuple1, tuple2, h=5):
        """
        kernel function (current function: (1 / math.sqrt(2 * math.pi) * 2 * h) * math.e ** (-distance / (2 * h ** 2))
        @param tuple1: [(type, latitude, longitude)]
        @param tuple2:
        @param h: bandwidth
        @return:
        """
        distance = 0
        if tuple1[0] == tuple2[0]:
            # distance = distance + dis_calculator.getNodeDistance(tuple1[1], tuple1[2], tuple2[1], tuple2[2])
            distance = distance + 1
        else:
            # distance = distance + 2 * dis_calculator.getNodeDistance(tuple1[1], tuple1[2], tuple2[1], tuple2[2])
            distance = distance + 0
        # return (1 / math.sqrt(2 * math.pi) * 2 * h) * math.e ** (-distance / (2 * h ** 2))
        return distance

    # translate the dataframe to tuple list
    task1 = task1.apply(lambda x: tuple(x), axis=1).values.tolist()
    task2 = task2.apply(lambda x: tuple(x), axis=1).values.tolist()
    count_distance = 0.0
    for tuple1 in task1:
        for tuple2 in task2:
            count_distance = count_distance + kernel_function(tuple1, tuple2)
    sim_s = count_distance / (len(task1) * len(task2))
    return sim_s


def learning_path_similarity_calculator(path1, path2):

    cos_sim = F.cosine_similarity(path1, path2, dim=1)
    return float(torch.mean(cos_sim))


def distribution_similarity_calculator(task1, task2):

    task1 = np.array(task1)
    task2 = np.array(task2)
    return ot.emd2([], [], ot.dist(task1, task2))


def load_similarity_cache(dataset, facter):
    global similarity_matrix_distribution
    global similarity_matrix_spatial
    global similarity_matrix_learning_path
    if dataset == "Porto_Grid":
        if facter == "distribution":
            similarity_matrix_distribution = np.loadtxt('similarity_matrix_distribution_grid.txt').tolist()
            print("load the distribution similarity matrix!", "Porto_Taxi_grid")
        if facter == "spatial":
            similarity_matrix_spatial = np.loadtxt('similarity_matrix_spatial_grid.txt').tolist()
            print("load the spatial similarity matrix!", "Porto_Taxi_grid")
            # print(similarity_matrix_spatial)
        if facter == "learning_path":
            similarity_matrix_learning_path = np.loadtxt('similarity_matrix_learning_path_grid.txt').tolist()
            print("load the learning_path similarity matrix!", "Porto_Taxi_grid")
        if facter == "learning_path_random":
            similarity_matrix_learning_path = np.loadtxt('similarity_matrix_learning_path_grid_random.txt').tolist()
            print("load the learning_path_prandom similarity matrix!", "Porto_Taxi_grid")



def similarity_calculator(task1, task2, facter,flag_s=True, flag_l=True, flag_d=True):


    m = 100
    nor_d_e_max = pow(m, 1)
    nor_d_e_min = pow(m, 0)
    nor_d = 500
    nor_l = 2
    if facter == "distribution":
        if similarity_matrix_distribution[task1.ID][task2.ID] == -1:
            if flag_d:
                sim_row = distribution_similarity_calculator(task1.data_set, task2.data_set)
                sim_d = (nor_d - sim_row) / nor_d
                sim_d = (pow(m, sim_d) - nor_d_e_min) / (nor_d_e_max - nor_d_e_min)
            else:
                sim_d = 0
            similarity_matrix_distribution[task1.ID][task2.ID] = similarity_matrix_distribution[task2.ID][
                task1.ID] = sim_d
        return similarity_matrix_distribution[task1.ID][task2.ID]

    elif facter == "spatial":
        if similarity_matrix_spatial[task1.ID][task2.ID] == -1:
            if flag_s:
                sim_s = spatial_similarity_calculator(task1.spatial_feature, task2.spatial_feature)
            else:
                sim_s = 0
            similarity_matrix_spatial[task1.ID][task2.ID] = similarity_matrix_spatial[task2.ID][
                task1.ID] = sim_s
        return similarity_matrix_spatial[task1.ID][task2.ID]

    elif facter == "learning_path":
        if similarity_matrix_learning_path[task1.ID][task2.ID] == -1:
            if flag_l:
                sim_l = (1 + learning_path_similarity_calculator(task1.k_step_learning_path,
                                                                 task2.k_step_learning_path)) / nor_l
            else:
                sim_l = 0
            similarity_matrix_learning_path[task1.ID][task2.ID] = similarity_matrix_learning_path[task2.ID][
                task1.ID] = sim_l
        return similarity_matrix_learning_path[task1.ID][task2.ID]

    elif facter == "learning_path_random":
        if similarity_matrix_learning_path[task1.ID][task2.ID] == -1:
            if flag_l:
                sim_l = (1 + learning_path_similarity_calculator(task1.k_step_learning_path,
                                                                 task2.k_step_learning_path)) / nor_l
            else:
                sim_l = 0
            similarity_matrix_learning_path[task1.ID][task2.ID] = similarity_matrix_learning_path[task2.ID][
                task1.ID] = sim_l
        return similarity_matrix_learning_path[task1.ID][task2.ID]



def generate_learning_path(data_path, sampling_step,  learning_task, dataset, model_name, k, args, flag="my"):
    """
    generate the learning path
    @param task:
    @param dataset:
    @param model_name:
    @param k:
    @param flag:
    @return:
    """

    if model_name == "LSTM_Encoder_Decoder":
        if flag == "my":
            target_path = "learning_path\\test\\"
        model = learning_task.learner

        if flag == "my":
            model_path = "distribution_spatial\\"
            task_tree = load_the_task_tree_DFS(model_path)
            parameter_path = traverse_old(task_tree, str(learning_task.ID) + ".txt")
            if parameter_path is not None:
                checkpoint = torch.load(parameter_path + "parameters.pth")
                model.module.load_state_dict(checkpoint)

        file_path = data_path
        for temp_name in os.listdir("temp_generate_learning_path\\data\\"):
            os.remove("temp_generate_learning_path\\data\\" + temp_name)
        shutil.copy2(file_path + str(learning_task.ID) + ".txt", "temp_generate_learning_path\\data\\" + str(learning_task.ID) + ".txt")

        if dataset == "Porto_Grid":
            x_spt, y_spt, x_qry, y_qry = Porto(
                "temp_generate_learning_path\\",
                iw=args.seq_in, ow=args.seq_out,
                s=args.stride, features=args.num_features,
                ratio=1).__read_data__()

            if not os.path.exists(target_path + str(learning_task.ID) + "\\"):
                os.makedirs(target_path + str(learning_task.ID) + "\\")
            model.train_model_learning_path_construction(x_spt[0], y_spt[0], n_epochs=k, target_len=args.seq_out,
                                                         batch_size=len(x_spt),
                                                         path=target_path + str(learning_task.ID) + "\\")


def generate_single_learning_path(learning_task, data_path, args, model, target_path, k=5):
    if args.model_name == "LSTM_Encoder_Decoder":
        if args.dataset_name == "Hannover":
            x_spt, y_spt, x_qry, y_qry = T_Drive_task_group(
                data_path,
                iw=args.seq_in, ow=args.seq_out,
                s=args.stride, features=args.num_features,
                ratio=1).__read_data__()
            if not os.path.exists(target_path + str(learning_task.ID) + "\\"):
                os.mkdir(target_path + str(learning_task.ID) + "\\")
            model.train_model_learning_path_construction(x_spt[0], y_spt[0], n_epochs=k, target_len=args.seq_out,
                                                         batch_size=len(x_spt),
                                                         path=target_path + str(learning_task.ID))


def generate_all_the_learning_path(data_path, sampling_step):
    file_list = os.listdir(data_path)
    user_list = []
    for file_name in file_list:
        user_list.append(int(file_name.split(".")[0]))
    args = argparser_return()
    model = lstm_seq2seq(input_size=args.num_features, hidden_size=args.hidden_size).to("cuda:0")

    maml = l2l.algorithms.MAML(model, args.update_lr)
    for user in user_list:
        generate_learning_path(data_path, sampling_step, LearningTask(user, maml, args, "test", ""), "Porto_Grid",
                               "LSTM_Encoder_Decoder", 5, args)
