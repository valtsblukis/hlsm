from collections import namedtuple
from typing import Callable, Union, Iterator, Tuple

from lgp.abcd.task import Task
from lgp.env.alfred.wrapping.annotations import AlfredAnnotations, TrajData

QUICK_DEBUG = False
QUICK_DEBUG_CUTOFF = 50


TASK_TYPES = [
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place"
]

# Lightweight class used to keep track of which tasks have already been done etc.
TaskRecord = namedtuple('Task', ['datasplit', 'task_id', 'repeat_idx'])


class AlfredTask(Task):
    def __init__(self, data_split: str, task_id: str, repeat_idx: int):
        super().__init__()
        self.traj_data: TrajData = AlfredAnnotations.load_traj_data_for_task(data_split, task_id)
        self.repeat_idx = repeat_idx
        self.data_split = data_split

    # This method is used to obtain the corresponding natural language phrase
    def __str__(self):
        return self.traj_data.get_task_description(self.repeat_idx)

    def is_test(self):
        # Returns True if this is a test example without plans / goal-condition / success data available
        return self.traj_data.is_test()

    def get_task_id(self):
        return self.traj_data.get_task_id()

    def get_repeat_idx(self):
        return self.repeat_idx

    def get_task_type(self):
        return self.traj_data.get_task_type()

    def get_data_split(self):
        return self.data_split

    def get_record(self) -> TaskRecord:
        return TaskRecord(self.get_data_split(), self.get_task_id(), self.get_repeat_idx())

    @classmethod
    def make_task_type_filter(cls, allowed_types):
        if len(allowed_types) == 0:
            task_filter = lambda m: True
        else:
            task_filter: Callable[["AlfredTask"], bool] = lambda m: m.get_task_type() in allowed_types
        return task_filter

    @classmethod
    def make_task_id_filter(cls, allowed_ids):
        if len(allowed_ids) == 0:
            task_filter = lambda m: True
        else:
            task_filter: Callable[["AlfredTask"], bool] = lambda m: m.get_task_id() in allowed_ids
        return task_filter

    @classmethod
    def iterate_all_tasks(cls,
                          data_splits=("train",),
                          task_filter: Union[None, Callable[["AlfredTask"], bool]] = None) -> Iterator[Tuple["AlfredTask", int]]:
        if task_filter is None:
            print("USING DEFAULT TASK FILTER - INCLUDE ALL TASKS")
            task_filter = lambda m: True

        count = 0
        alfred_annotations = AlfredAnnotations()
        print(f"Iterating tasks from: {alfred_annotations.splits_path}")
        for data_split in data_splits:
            all_task_ids = alfred_annotations.get_all_task_ids_in_split(data_split)
            for i, task_id in enumerate(all_task_ids):
                num_repeats = alfred_annotations.get_num_repeats(data_split, task_id)
                for repeat_idx in range(num_repeats):
                    task = AlfredTask(data_split, task_id, repeat_idx)
                    # Only yield tasks that pass the task filter
                    if task_filter(task):
                        print(f"INCLUDE {count:5d} : {task.get_task_id()}: {str(task)}:{repeat_idx}")
                        count += 1
                        yield task, count
                    else:
                        print(f"EXCLUDE {count:5d} : {task.get_task_id()}: {str(task)}:{repeat_idx}")
                        count += 1

                if QUICK_DEBUG and count > QUICK_DEBUG_CUTOFF:
                   break
        return



# Loop over all tasks - add any experimental statistics gathering here

def task_traj_lengths():
    import numpy as np

    tasks_by_type = []
    for task, cnt in AlfredTask.iterate_all_tasks():
        task_type = task.get_task_type()
        tasks_by_type.append((task_type, task))

    print_data = []

    for task_type in TASK_TYPES:
        filtered_tasks = [m[1] for m in filter(lambda m: m[0] == task_type, tasks_by_type)]

        from lgp.env.alfred.alfred_action import INTERACT_ACTION_TYPES


        def compute_hl_demo_length(task):
            actseq = task.traj_data.get_low_actions()
            actseq = [m['api_action']['action'] for m in actseq]
            actseq_f = list(filter(lambda m: m in INTERACT_ACTION_TYPES, actseq))
            return len(actseq_f) + 1


        hl_demo_lengths = [compute_hl_demo_length(task) for task in filtered_tasks]
        l = np.asarray(hl_demo_lengths)

        row = [task_type, f"{l.mean():.3f}", f"{l.std():.3f}", f"{np.percentile(l, 10):.3f}",
               f"{np.percentile(l, 90):.3f}"]
        print_data.append(row)
        #
        # print(f"Task: {task_type}, mean len: {l.mean()}, stddev len: {l.std()}")

    from tabulate import tabulate

    print("High-level action sequence length:")
    print(tabulate(print_data, headers=["Task Type", "Mean", "Stddev", "10th %ile", "90th %ile"]))


def extract_seen_scenes():
    scenes = []
    for task, _ in AlfredTask.iterate_all_tasks(data_splits=("valid_unseen",)):
        #assert task.data_split == "tests_unseen"
        scenes.append(task.traj_data.get_scene_number())
    scenes = list(sorted(set(scenes)))
    print("")
    print(scenes)


if __name__ == "__main__":
    extract_seen_scenes()
