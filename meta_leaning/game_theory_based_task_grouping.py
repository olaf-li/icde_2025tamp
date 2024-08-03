from similarity_calculation import *
import kmedoids
from meta_learning.task_tree import *
from meta_learning import lstm_encoder_decoder
import shutil

CLUSTER_ID = 0


def cluster_ID():
    global CLUSTER_ID
    CLUSTER_ID = CLUSTER_ID + 1
    return CLUSTER_ID


class LearningTaskCluster:
    def __init__(self, task_list, facter):

        self.ID = cluster_ID()
        self.facter = facter
        self.task_list = task_list
        self.Q = self.Q_calculator()

    def Q_calculator(self):
        Q_value = 0
        if len(self.task_list) > 1:
            for i in range(len(self.task_list)):
                for j in range(i + 1, len(self.task_list)):
                    Q_value = Q_value + 2 * similarity_calculator(self.task_list[i], self.task_list[j], self.facter)
            Q_value = Q_value / (len(self.task_list) * (len(self.task_list) - 1))
        else:
            Q_value = 0.1  # float("-inf")
        return Q_value

    def update_Q(self, new_task):

        increase_value = 0
        for i in range(len(self.task_list)):
            increase_value = increase_value + 2 * similarity_calculator(self.task_list[i], new_task, self.facter)
        return (self.Q * (len(self.task_list) * (len(self.task_list) - 1)) + increase_value) / (
                len(self.task_list) * (len(self.task_list) + 1))

    def add_new_task_with_value(self, new_task, value):

        self.task_list.append(new_task)
        self.Q = value

    def add_new_task(self, new_task):

        self.Q = self.update_Q(new_task)
        self.task_list.append(new_task)

    def remove_task(self, task):

        if len(self.task_list) > 2:
            self.task_list.remove(task)
            decrease_value = 0
            for i in range(len(self.task_list)):
                decrease_value = decrease_value + 2 * similarity_calculator(self.task_list[i], task, self.facter)
            self.Q = (self.Q * (len(self.task_list) * (len(self.task_list) + 1)) - decrease_value) / \
                     (len(self.task_list) * (len(self.task_list) - 1))
        else:

            self.task_list[0].cluster = None
            self.task_list.remove(self.task_list[0])

    def calculate_utility_for_task(self, task):

        decrease_value = 0
        for i in range(len(self.task_list)):
            if self.task_list[i].ID != task.ID:
                decrease_value = decrease_value + 2 * similarity_calculator(self.task_list[i], task, self.facter)
        if len(self.task_list) > 2:
            Q_before = (self.Q * (len(self.task_list) * (len(self.task_list) - 1)) - decrease_value) / \
                       ((len(self.task_list) - 1) * (len(self.task_list) - 2))
        else:
            Q_before = 0.1
        return self.Q - Q_before


def init_grouping_by_k_means(user_list, k, args):

    if k > len(user_list):
        k = len(user_list)
    distance_matrix, learning_task_list = distance_matrix_calculation(user_list, args)
    clustering_alg = kmedoids.KMedoids(k)
    c = clustering_alg.fit(distance_matrix)
    return c.labels_, learning_task_list


def best_response_framework(learning_task_list, cluster_list):

    count_k = 0
    Nash_equ_flag = 0
    while Nash_equ_flag == 0:
        count_k = count_k + 1
        Nash_equ_flag = 1
        for player in learning_task_list:
            max_utility = - 10000
            max_cluster = None
            for cluster in cluster_list:
                if len(cluster.task_list) == 0:
                    continue
                if cluster is not player.cluster:
                    utility = cluster.update_Q(player) - cluster.Q
                else:
                    utility = player.utility
                if utility > max_utility:
                    max_utility = utility
                    max_cluster = cluster

            if max_cluster is not player.cluster:
                Nash_equ_flag = 0

                if player.cluster is not None:
                    player.cluster.remove_task(player)

                max_cluster.add_new_task_with_value(player, max_utility + max_cluster.Q)
                player.cluster = max_cluster
                player.utility = max_utility

    return cluster_list


def distance_matrix_calculation(user_list, args):

    def find_learning_task_by_user_id(user_id):
        for i in range(len(learning_task_list)):
            if user_id == learning_task_list[i].ID:
                return learning_task_list[i]

    distance_matrix = []
    learning_task_list = []
    for user in user_list:
        if args.model_name == "LSTM_Encoder_Decoder":
            meta_model = lstm_encoder_decoder.lstm_seq2seq(input_size=args.seq_in, hidden_size=args.hidden_size)
        else:
            meta_model = lstm_encoder_decoder.lstm_seq2seq(input_size=10, hidden_size=15)
        learning_task_list.append(LearningTask(user, meta_model, args, 10, "train", dataset=args.dataset_name))
        temp_list = []
        for i in range(len(user_list)):
            temp_list.append(0)
        distance_matrix.append(temp_list)
    for i in range(len(user_list)):
        for j in range(i, len(user_list)):
            temp_sim = similarity_calculator(
                find_learning_task_by_user_id(user_list[i]), find_learning_task_by_user_id(user_list[j]), args.facter)
            if temp_sim == 0.0:
                distance_matrix[i][j] = distance_matrix[j][i] = float('inf')
            else:
                distance_matrix[i][j] = distance_matrix[j][i] = 1 / temp_sim

    return distance_matrix, learning_task_list


def run(user_list, args, k):

    label_list, learning_task_list = init_grouping_by_k_means(user_list, k, args)
    task_cluster_list = []
    for i in range(k):
        task_cluster_list.append([])
    for i in range(len(label_list)):
        task_cluster_list[label_list[i]].append(learning_task_list[i])
    cluster_list = []
    for i in range(k):
        if len(task_cluster_list[i]) < 1:
            continue
        else:
            temp_cluster = LearningTaskCluster(task_cluster_list[i], args.facter)
            cluster_list.append(temp_cluster)
            for learning_task in task_cluster_list[i]:
                learning_task.cluster = temp_cluster
                learning_task.utility = 0
    for cluster in cluster_list:
        for task in cluster.task_list:
            task.utility = cluster.calculate_utility_for_task(task)
    i = 1
    for cluster in cluster_list:
        print(i, len(cluster.task_list))
        i = i + 1
    return cluster_list


def construct_task_tree(data_path, target_path, args):

    user_list = []
    file_list = os.listdir(data_path)
    for file in file_list:
        user_list.append(int(file.split(".")[0]))

    load_similarity_cache(args.dataset_name, args.facter)
    cluster_list = []
    if args.facter == "distribution":
        k = 20
    elif args.facter == "spatial":
        k = 5
    elif args.facter == "learning_path":
        k = 5
    elif args.facter == "learning_path_pami":
        k = 20
    elif args.facter == "learning_path_random":
        k = 20

    if k <= 1:
        return


    cluster_list_1 = run(user_list, args, k)

    ID_check_list = []
    for cluster in cluster_list_1:
        for task in cluster.task_list:
            if task.ID not in ID_check_list:
                ID_check_list.append(task.ID)
            else:
                print("there is an error! ", cluster.ID, task.ID)

    if len(cluster_list_1) <= 1:
        return
    i = 1
    print(len(cluster_list_1))
    if args.facter == "distribution":
        for cluster_1 in cluster_list_1:
            if len(cluster_1.task_list) < 1:
                continue
            # print(cluster_1.Q)
            if cluster_1.Q < 0.99:
                # print(target_path + "m" + str(i) + "\\" + "data\\")
                print("m" + str(i))
            # i = i + 1
            # continue
            if not os.path.exists(target_path + "m" + str(i) + "\\"):
                os.mkdir(target_path + "m" + str(i) + "\\")

            cluster_list.append(cluster_1)
            if not os.path.exists(target_path + "m" + str(i) + "\\" + "data\\"):
                os.mkdir(target_path + "m" + str(i) + "\\" + "data\\")
            for task in cluster_1.task_list:
                shutil.copy(data_path + str(task.ID) + '.txt', target_path + "m" + str(i) + "\\" + "data\\")
            i = i + 1

    elif args.facter == "spatial":
        for cluster_1 in cluster_list_1:
            if len(cluster_1.task_list) < 1:
                continue
            print(cluster_1.Q)
            if cluster_1.Q < 0.2:
                # print(target_path + "m" + str(i) + "\\" + "data\\")
                print("m" + str(i))
            # i = i + 1
            # continue
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\")

            cluster_list.append(cluster_1)
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            for task in cluster_1.task_list:
                shutil.copy(data_path + str(task.ID) + '.txt', target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            i = i + 1

    elif args.facter == "learning_path":
        for cluster_1 in cluster_list_1:
            if len(cluster_1.task_list) < 1:
                continue
            print(cluster_1.Q)
            if cluster_1.Q < 0.25:
                # print(target_path + "m" + str(i) + "\\" + "data\\")
                print("m" + str(i))
            # i = i + 1
            # continue
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\")

            cluster_list.append(cluster_1)
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            for task in cluster_1.task_list:
                shutil.copy(data_path + str(task.ID) + '.txt', target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            i = i + 1

    elif args.facter == "learning_path_random":
        for cluster_1 in cluster_list_1:
            if len(cluster_1.task_list) < 1:
                continue
            print(cluster_1.Q)
            if cluster_1.Q < 0.25:
                # print(target_path + "m" + str(i) + "\\" + "data\\")
                print("m" + str(i))
            # i = i + 1
            # continue
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\")

            cluster_list.append(cluster_1)
            if not os.path.exists(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\"):
                os.mkdir(target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            for task in cluster_1.task_list:
                shutil.copy(data_path + str(task.ID) + '.txt', target_path + target_path.split("\\")[-2] + str(i) + "\\" + "data\\")
            i = i + 1

