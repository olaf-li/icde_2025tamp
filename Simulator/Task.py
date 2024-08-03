class Task:
    def __init__(self, lo, le, t, at, ID):

        self.pick_up = lo
        self.drop_off = le
        self.deadline = t
        self.arrive_time = at
        self.ID = ID
        self.reject_num = 0


def task_init_beijing():

    tasks = []
    task_file_path = "tasks.txt"
    with open(task_file_path) as file:
        for line in file:
            strings = line.replace("\n", "").split("\t")
            lo = [float(strings[1]), float(strings[0])]
            le = [float(strings[3]), float(strings[2])]
            arrive_time = strings[4]
            deadline = strings[5]
            tasks.append(Task(lo, le, arrive_time, deadline))
    print("the read of tasks finished; ", len(tasks), " tasks have been read")
    return tasks

def read_task_Porto(flag):
    id = 0
    tasks = []
    if flag == "test":
        task_file_path = "task\\tasks.txt"
    with open(task_file_path) as file:
        for line in file:
            strings = line.replace("\n", "").split(",")
            lo = [float(strings[2]), float(strings[3])]
            le = [float(strings[2]), float(strings[3])]
            arrive_time = int(strings[0])
            deadline = int(strings[1])
            tasks.append(Task(lo, le, deadline, arrive_time, id))
            id = id + 1
    return tasks


def task_init(dataset="Hannover"):
    if dataset == "Porto_Grid":
        return read_task_Porto(flag="task")

