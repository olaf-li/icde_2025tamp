import os
import learn2learn as l2l
from geopy.distance import geodesic

time_stride = 30
RADIUS = 6000


class Worker:
    def __init__(self, ID, r, mode, c, d, pr, at, model, data_path, speed, matching_rate):
        self.ID = ID
        self.trajectory = r
        self.trajectory_mode = mode
        self.radius = d
        self.pre_trajectory = pr
        self.prediction_trajectory = pr
        self.arrive_time = at
        self.state = 0
        self.speed_list = speed
        self.current_location = (self.trajectory[0][1], self.trajectory[0][2])
        self.current_segment_index = 0
        self.current_location_index = 0
        self.current_task = [(0, 0), (0, 0), -1, -1]
        self.model = model
        self.data_path = data_path

        self.task_state = self.current_location_index

        self.current_assigned_tasks = {}

        self.unavailable_tasks = set()

        self.calculated_tasks = {}

        self.candidate_set = {}

        self.matching_rate = matching_rate

    def move_forward(self, current_time, move_time):
        """

        @param current_time:
        @param move_time:
        @return:
        """
        if self.state == 0:
            if current_time + move_time > self.trajectory[self.current_segment_index][-1][0]:
                """ 说明工人已经要走完当前的segment了 """
                self.current_segment_index = self. current_segment_index + 1
                if self.current_segment_index >= len(self.trajectory):
                    """ 工人已经永久离开平台，将状态设置为3 leave the platform """
                    self.state = 3
                else:
                    if self.trajectory[self.current_segment_index][0][0] > current_time + move_time:
                        """ 工人还未开始下一次在线，将状态设置为2 offline """
                        self.state = 2
                        self.current_location_index = 0
                    else:
                        """ 工人在批次结束后已经进入下一个在线轮次，将状态设置为0 online """
                        self.state = 0
                        """ 更新工人当前的位置 """
                        current_track = self.trajectory[self.current_segment_index]
                        flag_bug = 0
                        for i in range(0, len(current_track) - 1):

                            if current_track[i][0] <= current_time + move_time < current_track[i + 1][0]:
                                """ 找到批次结束后工人所在的位置 """
                                self.current_location_index = i
                                flag_bug = 1
                        if flag_bug == 0:
                            print(self.ID)
                            print(self.trajectory[self.current_segment_index - 1])
                            print(self.trajectory[self.current_segment_index])
                            print(current_time, move_time, current_time + move_time)
                            print(" when state == 0, there is a bug in terms of function: Worker.move_forward !")
            else:
                """ 什么都不做 """

        elif self.state == 1:
            """ 当工人正在执行任务时 """
            if current_time + move_time > self.trajectory[self.current_segment_index][-1][0]:
                """ 表示工人已经完成了当前分配的任务 """
                self.current_segment_index = self.current_segment_index + 1
                if self.current_segment_index >= len(self.trajectory):
                    """ 工人已经永久离开平台，将状态设置为3 leave the platform """
                    self.state = 3
                else:
                    if self.trajectory[self.current_segment_index][0][0] > current_time + move_time:
                        """ 工人还未开始下一次在线，将状态设置为2 offline """
                        self.state = 2
                        self.current_location_index = 0
                    else:
                        """ 工人在批次结束后已经进入下一个在线轮次，将状态设置为0 online """
                        self.state = 0
                        """ 更新工人当前的位置 """
                        current_track = self.trajectory[self.current_segment_index]
                        flag_bug = 0
                        for i in range(0, len(current_track) - 1):
                            if current_track[i][0] <= current_time + move_time < current_track[i + 1][0]:
                                """ 找到批次结束后工人所在的位置 """
                                self.current_location_index = i
                                flag_bug = 1
                        if flag_bug == 0:
                            print(self.ID)
                            print(self.trajectory[self.current_segment_index - 1])
                            print(self.trajectory[self.current_segment_index])
                            print(current_time, current_time + move_time)
                            print(" when state == 1, there is a bug in terms of function: Worker.move_forward !")
            else:
                """ 什么都不做 """

        elif self.state == 2:
            if current_time + move_time >= self.trajectory[self.current_segment_index][self.current_location_index][0]:
                """ 工人已经上线，将状态设置为0 online"""
                self.state = 0
                """ 更新工人当前的位置 """
                current_track = self.trajectory[self.current_segment_index]
                flag_bug = 0
                for i in range(0, len(current_track) - 1):
                    if current_track[i][0] <= current_time + move_time < current_track[i + 1][0]:
                        """ 找到批次结束后工人所在的位置 """
                        self.current_location_index = i
                        flag_bug = 1
                if flag_bug == 0:
                    print(self.ID)
                    print(self.trajectory[self.current_segment_index - 1])
                    print(self.trajectory[self.current_segment_index])
                    print(current_time, current_time + move_time)
                    print(" when state == 2, there is a bug in terms of function: Worker.move_forward !")
            else:
                """ 什么都不做 """

        # move_index = int(move_time / time_stride)
        # if self.current_location_index + move_index > len(self.trajectory):
        #     """
        #     工人的行程已经结束了，工人将会离开平台
        #     """
        #     self.state = 2
        # else:
        #     self.current_location_index = self.current_location_index + move_index
        #     if self.current_location_index > self.task_state:
        #         self.task_state = self.current_location_index
        #         self.state = 0

        # if self.task_state == 0:
        #     current_time = self.trajectory[self.current_location_index][0]
        #     if len(self.assigned_tasks) != 0:
        #         for tuple in self.assigned_tasks:
        #             if tuple[1] < current_time + move_time:
        #                 """
        #                 说明在这个批次内要开始执行任务了
        #                 将任务开始点之后的轨迹的时间后延
        #                 """
        #                 for i in range(tuple[0], len(self.trajectory)):
        #                     self.trajectory[i][0] = self.trajectory[i][0] + tuple[2] - tuple[1]
        #                 if tuple[2] < current_time + move_time:
        #                     """
        #                     说明任务可在当前时间片内执行完成
        #                     按照这种情况，更行工人的位置信息
        #                     """
        #                     move_index = int((move_time + current_time - (tuple[2] - tuple[1])) / 2)
        #                     if self.current_location_index + move_index < len(self.trajectory):
        #                         self.current_location_index = self.current_location_index + move_index
        #                     else:
        #                         """
        #                         工人的行程已经结束了，工人将会离开平台
        #                         """
        #                         self.task_state = 2
        #                 else:
        #                     """
        #                     说明这个批次内任务无法执行完成
        #                     """
        #                     self.task_state = 2
        #
        # else:
        #
        #
        #
        # if self.current_location_index + move_index < len(self.trajectory):
        #     self.current_location_index = self.current_location_index + move_index
        # else:

    def get_current_location(self, time_stamp):

        for i in range(len(self.trajectory)):
            if time_stamp - self.trajectory[i][0] <= time_stride:
                return i


def worker_init(dataset="Porto_Grid", sim_type="real"):

    if dataset == "Porto_Grid":
        return worker_init_Porto(sim_type)


def worker_init_Porto(flag):

    radius = RADIUS

    args = argparser_return()

    workers = []

    file_path = "test\\"
    file_list = os.listdir(file_path)
    pre_file_path = "workers\\"

    for name in file_list:
        track = []  # [ [ [timestamp, lat, long] ] ]
        speed_list = []
        track_point = []
        with open(file_path + name) as file:
            for line in file:
                strings = line.replace("\n", "").split(",")
                if len(strings) == 1:
                    if strings[0] == "END":
                        track.append(track_point)
                        distance = 0
                        for i in range(1, len(track_point)):
                            distance = distance + geodesic((track_point[i - 1][1], track_point[i - 1][2]),
                                                           (track_point[i][1], track_point[i][2])).meters
                        speed_list.append(distance / (time_stride * len(track_point)))
                        track_point = []
                    else:
                        continue
                else:
                    track_point.append([int(strings[0]), float(strings[1]), float(strings[2])])

        pre_track = []
        pre_track_point = []
        with open(pre_file_path + name) as file:
            for line in file:
                strings = line.replace("\n", "").split(",")
                if len(strings) == 1:
                    if strings[0] == "END":
                        pre_track.append(pre_track_point)
                        distance = 0
                        for i in range(1, len(pre_track_point)):
                            distance = distance + geodesic((pre_track_point[i - 1][1], pre_track_point[i - 1][2]),
                                                           (pre_track_point[i][1], pre_track_point[i][2])).meters
                        pre_track_point = []
                    else:
                        continue
                else:
                    lat_grid = int(round(float(strings[1])))
                    long_grid = int(round(float(strings[2])))
                    this_lat, this_long = map_grid_to_gps(lat_grid, long_grid)
                    pre_track_point.append([int(strings[0]), this_lat, this_long])

        ID = int(name.split(".")[0])
        arrive_time = track[0][0][0]
        this_model = l2l.algorithms.MAML(lstm_seq2seq(input_size=args.num_features, hidden_size=args.hidden_size),
                                         args.update_lr)
        data_path = file_path + name
        if flag == "real":
            prediction_track = track
        else:
            prediction_track = pre_track
        matching_rate_list = []
        for i in range(len(track)):
            for j in range(len(track[i])):
                real_location = (track[i][j][1], track[i][j][2])
                pre_location = (prediction_track[i][j][1], track[i][j][2])
                if geodesic(real_location, pre_location).meters <= radius / 2:
                    matching_rate_list.append(1)
                else:
                    matching_rate_list.append(0)

        workers.append(Worker(ID, track, "GPS", None, radius, prediction_track, arrive_time, this_model,
                              data_path, speed_list, sum(matching_rate_list) / len(matching_rate_list)))

    return workers
