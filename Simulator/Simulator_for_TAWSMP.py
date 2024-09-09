import time

from Simulator.Worker import worker_init
from Simulator.Task import task_init
from task_assignment.task_assignment import Batch_Task_Assignment, calculate_weight_assign
from similarity_calculation import *

batch_size = 120  # 匹配的批次大小，单位：秒


class Simulator:

    def __init__(self, w_n, w_n_path, t, e, simulate_step=2, batch_size=120, sim_type="real"):

        self.n = w_n
        self.args = argparser_return()
        self.worker_path = w_n_path
        self.current_worker_index = 0
        self.total_workers = worker_init(dataset=self.args.dataset_name, sim_type=sim_type)
        self.sim_type = sim_type
        self.current_task_index = 0
        self.total_tasks = task_init(dataset=self.args.dataset_name)
        self.batch_size = batch_size
        self.begin_time = t
        self.end_time = e
        self.simulate_step = simulate_step
        self.model_path = "root\\"
        self.task_tree = load_the_task_tree_DFS(self.model_path)

    def get_worker_by_ID(self, ID):
        """

        @param ID:
        @return:
        """
        for worker in self.total_workers:
            if worker.ID == ID:
                return worker
        return None

    def get_workers_tasks_of_next_batch(self, begin_time, end_time):
        """
        get workers and tasks of a batch
        @param begin_time:
        @param end_time:
        @return:
        """
        tasks = []
        workers = []
        for task in self.total_tasks:
            if begin_time <= task.arrive_time < end_time:
                tasks.append(task)
        #
        for worker in self.total_workers:
            if begin_time <= worker.arrive_time < end_time:
                workers.append(worker)
        return tasks, workers

    def run_matching(self):

        test_time_List = []
        test_List = []

        total_cost = 0
        count_reject = 0
        count_total = 0

        real_or_pre = self.sim_type

        tasks = []
        online_workers = []  # state == 0
        unavailable_workers = []  # state == 1, 2
        departing_workers = []  # state == 3

        number_of_assigned_task = 0
        number_of_current_assigned_task = 0
        number_of_total_workers = 0
        number_of_total_tasks = 0
        running_time = 0

        for t in range(self.begin_time, self.end_time, self.batch_size):

            # read workers and tasks arrived at the current batch
            new_tasks, new_workers = self.get_workers_tasks_of_next_batch(t, t + self.batch_size)
            number_of_total_workers = number_of_total_workers + len(new_workers)
            number_of_total_tasks = number_of_total_tasks + len(new_tasks)

            test_time_List.append(t)
            test_List.append(len(new_workers))
            number_of_current_assigned_task = 0

            tasks.extend(new_tasks)
            online_workers.extend(new_workers)

            time1 = time.time()
            T_A = Batch_Task_Assignment(online_workers, tasks)
            matching_plan = T_A.KM(real_or_pre)

            assigned_workers = []
            assigned_tasks = []
            count_total = count_total + len(matching_plan)
            for matching in matching_plan:

                this_task = matching[1]
                this_worker = matching[0]

                this_flag, detour = calculate_weight_assign(this_task, this_worker)
                if this_flag == -1:
                    this_task.reject_num = this_task.reject_num + 1
                    continue
                else:
                    total_cost = total_cost + detour

                    if this_worker.current_segment_index >= len(this_worker.trajectory) - 1:
                        continue

                    speed = this_worker.speed_list[this_worker.current_segment_index]
                    time_delay = int(detour / speed)
                    if time_delay + this_worker.trajectory[this_worker.current_segment_index][-1][0] > \
                        this_worker.trajectory[this_worker.current_segment_index + 1][0][0]:

                        this_worker.trajectory[this_worker.current_segment_index][-1][0] = \
                            this_worker.trajectory[this_worker.current_segment_index][-1][0] + time_delay
                        for i in range(this_worker.current_segment_index + 1, len(this_worker.trajectory)):
                            this_time_delay = this_worker.trajectory[i][0][0] - this_worker.trajectory[i - 1][-1][0]
                            if this_time_delay > 0:
                                for j in range(0, len(this_worker.trajectory[i])):
                                    this_worker.trajectory[i][j][0] = this_worker.trajectory[i][j][0] + this_time_delay
                            else:
                                break

                    this_worker.state = 1
                    assigned_workers.append(matching[0])
                    assigned_tasks.append(matching[1])
            running_time = running_time + time.time() - time1
            for task in assigned_tasks:
                number_of_assigned_task = number_of_assigned_task + 1
                number_of_current_assigned_task = number_of_current_assigned_task + 1
                tasks.remove(task)
            for worker in assigned_workers:
                online_workers.remove(worker)

            unavailable_workers.extend(assigned_workers)

            remove_workers_online = []
            remove_workers_unavailable = []
            add_workers_online = []
            add_workers_unavailable = []
            add_departing_workers = []
            for worker in online_workers:
                worker.move_forward(t, self.batch_size)
                if worker.state == 2:
                    remove_workers_online.append(worker)
                    add_workers_unavailable.append(worker)
                elif worker.state == 3:
                    remove_workers_online.append(worker)
                    add_departing_workers.append(worker)

            for worker in unavailable_workers:
                worker.move_forward(t, self.batch_size)
                if worker.state == 0:
                    remove_workers_unavailable.append(worker)
                    add_workers_online.append(worker)
                elif worker.state == 3:
                    remove_workers_unavailable.append(worker)
                    add_departing_workers.append(worker)

            for worker in remove_workers_online:
                online_workers.remove(worker)
            online_workers.extend(add_workers_online)

            for worker in remove_workers_unavailable:
                unavailable_workers.remove(worker)
            unavailable_workers.extend(add_workers_unavailable)

            departing_workers.extend(add_departing_workers)

            departing_tasks = []
            for task in tasks:
                if t >= task.deadline:
                    departing_tasks.append(task)
            for task in departing_tasks:
                tasks.remove(task)
