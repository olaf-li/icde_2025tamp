"""
given a batch of tasks and workers, it finds a solution with selected algorithms
"""
import subprocess

from geopy.distance import geodesic

BACK_LENGTH = 1
PRE_LENGTH = PRE_LENGTH


def calculate_weight_matching(task, worker, track):
    location = task.pick_up
    min_distance = float("inf")
    min_time = 0
    for item in track:
        dist = geodesic(location, (item[1], item[2])).meters
        if dist < min_distance:
            min_distance = dist
            min_time = item[0]
    if min_distance < worker.radius and min_distance / worker.speed_list[worker.current_segment_index] + min_time < task.deadline:
        return 1, min_distance
    else:
        return -1, min_distance


def calculate_weight_assign(task, worker):

    location = task.pick_up
    current_segment = worker.trajectory[worker.current_segment_index]
    current_Location = (current_segment[worker.current_location_index][1], current_segment[worker.current_location_index][2])
    next_must_location = (current_segment[-1][1], current_segment[-1][2])
    dis_1 = geodesic(current_Location, location).meters
    detour = dis_1 + geodesic(location, next_must_location).meters - geodesic(current_Location, next_must_location).meters

    if worker.current_segment_index < len(worker.trajectory) - 1:
        speed = worker.speed_list[worker.current_segment_index]
        time_delay = int(detour / speed)
        if time_delay + worker.trajectory[worker.current_segment_index][-1][0] > worker.trajectory[worker.current_segment_index + 1][-1][0]:
            # 表明会耽误工人接下来的行程
            return -1, detour
    task_time_delay = dis_1 / worker.speed_list[worker.current_segment_index]
    current_time = worker.trajectory[worker.current_segment_index][worker.current_location_index][0]
    if detour < worker.radius and task_time_delay + current_time < task.deadline:
        return 1, detour
    else:
        return -1, detour


class Batch_Task_Assignment:
    def __init__(self, workers, tasks):
        # the available workers
        self.workers = workers
        # the tasks to be assigned
        self.tasks = tasks

    def generate_bipartite_graph(self, flag):

        BG = [[0, 0, 0]]
        list_task = []
        list_worker = []
        task_id = 0
        worker_id = 0
        for i in range(len(self.workers)):
           for j in range(len(self.tasks)):
                current_trajectory_segment = self.workers[i].current_segment_index
                current_trajectory_index = self.workers[i].current_location_index
                if flag == "real":
                    track_length = len(self.workers[i].trajectory[current_trajectory_segment])
                    track = self.workers[i].trajectory[current_trajectory_segment][current_trajectory_index: min(current_trajectory_index + 5, track_length)]
                else:
                    """ flag == pre or no pre"""
                    back_length = BACK_LENGTH
                    pre_length = PRE_LENGTH
                    if self.workers[i].current_location_index >= back_length - 1:
                        real_track = self.workers[i].trajectory[current_trajectory_segment][current_trajectory_index - back_length: current_trajectory_index]
                    else:
                        real_track = self.workers[i].trajectory[current_trajectory_segment][0: current_trajectory_index]
                    track_length = len(self.workers[i].pre_trajectory[current_trajectory_segment])
                    track = real_track + self.workers[i].pre_trajectory[current_trajectory_segment][
                            current_trajectory_index: min(current_trajectory_index + pre_length, track_length)]

                if flag == "real":
                    result = calculate_weight_assign(self.tasks[j], self.workers[i])
                elif flag == "no_pre":
                    result = calculate_weight_matching(self.tasks[j], self.workers[i], [])
                else:
                    result = calculate_weight_matching(self.tasks[j], self.workers[i], track)

                if result[0] == 1:
                    if i not in list_worker:
                        worker_id = worker_id + 1
                        list_worker.append(i)
                        worker_index = worker_id - 1
                    else:
                        worker_index = list_worker.index(i)

                    if j not in list_task:
                        task_id = task_id + 1
                        list_task.append(j)
                        task_index = task_id - 1
                    else:
                        task_index = list_task.index(j)
                    BG.append([worker_index + 1, task_index + 1, int(12000 - result[1])])
        BG[0][0] = len(list_worker)
        BG[0][1] = len(list_task)
        BG[0][2] = len(BG) - 1
        return BG, list_worker, list_task

    def KM(self, flag):

        M = []
        workers = self.workers
        tasks = self.tasks
        if len(workers) == 0 or len(tasks) == 0:
            return M
        BG, worker_index_list, task_index_list = self.generate_bipartite_graph(flag)

        # print("\t生成二分图！")

        # print(BG)
        bg_argv = ""
        for i in range(len(BG)):
            bg_argv = bg_argv + str(BG[i][0]) + " " + str(BG[i][1]) + " " + str(BG[i][2]) + " "
        args = [".\KM.exe", bg_argv]

        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        output = stdout.decode()
        error = stderr.decode()
        number_of_assignment = int(output.split("\n")[0])
        if number_of_assignment == 0:
            return M
        assignment = output.split("\n")[1].split(" ")
        for i in range(len(assignment)):
            if assignment[i] == '':
                continue
            if assignment[i] != "0":
                this_worker = workers[worker_index_list[i]]
                this_task = tasks[task_index_list[int(assignment[i]) - 1]]
                M.append([this_worker, this_task])
        return M

    def generate_bipartite_graph_selected(self, flag, tasks, workers):

        BG = [[0, 0, 0]]
        list_task = []
        list_worker = []
        task_id = 0
        worker_id = 0
        for i in range(len(workers)):
            for j in range(len(tasks)):
                current_trajectory_segment = workers[i].current_segment_index
                current_trajectory_index = workers[i].current_location_index
                if flag == "real":
                    track_length = len(workers[i].trajectory[current_trajectory_segment])
                    track = workers[i].trajectory[current_trajectory_segment][current_trajectory_index: min(current_trajectory_index + 5, track_length)]
                else:
                    """ flag == pre """
                    back_length = BACK_LENGTH
                    pre_length = PRE_LENGTH
                    if workers[i].current_location_index >= back_length - 1:
                        real_track = workers[i].trajectory[current_trajectory_segment][current_trajectory_index - back_length: current_trajectory_index]
                    else:
                        real_track = workers[i].trajectory[current_trajectory_segment][0: current_trajectory_index]
                    track_length = len(workers[i].pre_trajectory[current_trajectory_segment])
                    track = real_track + workers[i].pre_trajectory[current_trajectory_segment][
                            current_trajectory_index: min(current_trajectory_index + pre_length, track_length)]

                if flag == "real":
                    result = calculate_weight_assign(tasks[j], workers[i])
                elif flag == "no_pre":
                    result = calculate_weight_matching(tasks[j], workers[i], [])
                else:
                    result = calculate_weight_matching(tasks[j], workers[i], track)
                if result[0] == 1:
                    if i not in list_worker:
                        worker_id = worker_id + 1
                        list_worker.append(i)
                        worker_index = worker_id - 1
                    else:
                        worker_index = list_worker.index(i)

                    if j not in list_task:
                        task_id = task_id + 1
                        list_task.append(j)
                        task_index = task_id - 1
                    else:
                        task_index = list_task.index(j)
                    BG.append([worker_index + 1, task_index + 1, int(12000 - result[1])])
        BG[0][0] = len(list_worker)
        BG[0][1] = len(list_task)
        BG[0][2] = len(BG) - 1
        return BG, list_worker, list_task

    def PPI(self, flag):
        """

        @param flag:
        @return:
        """
        epson = 5

        M = []
        workers = self.workers
        tasks = self.tasks

        assigned_tasks = []
        assigned_workers = []

        if len(workers) == 0 or len(tasks) == 0:
            return M
        M_B = []
        c_tasks = set()
        c_workers = set()
        for task in tasks:
            for worker in workers:
                current_trajectory_segment = worker.current_segment_index
                current_trajectory_index = worker.current_location_index
                if flag == "real":
                    track_length = len(worker.trajectory[current_trajectory_segment])
                    track = worker.trajectory[current_trajectory_segment][
                            current_trajectory_index: min(current_trajectory_index + 5, track_length)]
                else:
                    """ flag == pre """
                    back_length = BACK_LENGTH
                    pre_length = PRE_LENGTH
                    if worker.current_location_index >= back_length - 1:
                        real_track = worker.trajectory[current_trajectory_segment][
                                     current_trajectory_index - back_length: current_trajectory_index]
                    else:
                        real_track = worker.trajectory[current_trajectory_segment][0: current_trajectory_index]
                    track_length = len(worker.pre_trajectory[current_trajectory_segment])
                    track = real_track + worker.pre_trajectory[current_trajectory_segment][
                                         current_trajectory_index: min(current_trajectory_index + pre_length,
                                                                       track_length)]
                B = []
                for point in track:
                    this_timestamp = point[0]
                    this_location = (point[1], point[2])
                    this_distance = geodesic(this_location, task.pick_up).meters
                    speed = worker.speed_list[worker.current_segment_index]
                    d_t = (task.deadline - this_timestamp) * speed
                    threshold_distance = min(worker.radius / 2, d_t)
                    if this_distance + worker.radius / 4 < threshold_distance:
                        B.append(this_distance)
                if len(B) * worker.matching_rate >= 1:
                    c_tasks.add(task)
                    c_workers.add(worker)
                else:
                    M_B.append((B, task, worker, len(B) * worker.matching_rate))

        def this_KM(flag, this_tasks, this_workers):
            temp_M = []
            BG, worker_index_list, task_index_list = self.generate_bipartite_graph_selected(flag, this_tasks, this_workers)
            bg_argv = ""
            for i in range(len(BG)):
                bg_argv = bg_argv + str(BG[i][0]) + " " + str(BG[i][1]) + " " + str(BG[i][2]) + " "
            args = [".\KM.exe", bg_argv]

            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            output = stdout.decode()
            error = stderr.decode()

            number_of_assignment = int(output.split("\n")[0])
            if number_of_assignment == 0:
                return temp_M
            assignment = output.split("\n")[1].split(" ")
            for i in range(len(assignment)):
                if assignment[i] == '':
                    continue
                if assignment[i] != "0":
                    this_worker = this_workers[worker_index_list[i]]
                    this_task = this_tasks[task_index_list[int(assignment[i]) - 1]]
                    temp_M.append([this_worker, this_task])
            return temp_M

        M = this_KM(flag, list(c_tasks), list(c_workers))
        for matching in M:
            assigned_workers.append(matching[0])
            assigned_tasks.append(matching[1])
        c_tasks = set()
        c_workers = set()

        counter = 0
        M_B.sort(key=lambda x: x[3], reverse=True)
        for i in range(0, len(M_B)):
            if len(M_B[i][0]) > 0:
                if M_B[i][1] in assigned_tasks:
                    continue
                if M_B[i][2] in assigned_workers:
                    continue
                c_tasks.add(M_B[i][1])
                c_workers.add(M_B[i][2])
                counter = counter + 1
            else:
                M_f = this_KM(flag, list(c_tasks), list(c_workers))
                c_tasks = set()
                c_workers = set()
                for matching in M_f:
                    assigned_workers.append(matching[0])
                    assigned_tasks.append(matching[1])
                M = M + M_f
                break
            if counter == epson:
                M_f = this_KM(flag, list(c_tasks), list(c_workers))
                for matching in M_f:
                    assigned_workers.append(matching[0])
                    assigned_tasks.append(matching[1])
                M = M + M_f
                counter = 0
                c_tasks = set()
                c_workers = set()
        for task in self.tasks:
            if task not in assigned_tasks:
                c_tasks.add(task)
        for worker in self.workers:
            if worker not in assigned_workers:
                c_workers.add(worker)
        if len(c_tasks) != 0:
            M_f = this_KM(flag, list(c_tasks), list(c_workers))
            M = M + M_f
            for matching in M_f:
                assigned_workers.append(matching[0])
                assigned_tasks.append(matching[1])
        unassigned_tasks = []
        unassigned_workers = []
        for task in self.tasks:
            if task not in assigned_tasks:
                unassigned_tasks.append(task)
        for worker in self.workers:
            if worker not in assigned_workers:
                unassigned_workers.append(worker)
        M_f = this_KM(flag, unassigned_tasks, unassigned_workers)
        M = M + M_f
        return M