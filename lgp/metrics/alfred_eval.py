# This code is a functional copy of:
# https://github.com/askforalfred/alfred/tree/master/models/eval
# but with the following assumptions REMOVED:
#   - Agents are pytorch models
#   - ResNet features are used
#   - Training data is stored offline and pre-processed by the model
#   - Multithreading is used for parallel evaluation

# Instead, the following structure is NEW:
#   - Eval code works on pre-collected rollouts from a rollout actor (that can be spawned in parallel)

from typing import List, Dict
from lgp.env.alfred.wrapping.annotations import TrajData
from lgp.env.alfred.tasks import AlfredTask


class AlfredResults:
    def __init__(self):
        self.successes = []
        self.failures = []
        self.results = {}

    def printout(self):
        from pprint import pprint
        for type, results in self.results.items():
            print(f"TASK TYPE: {type}")
            pprint(results)


def get_metrics(successes, failures):
    '''
    compute overall succcess and goal_condition success rates along with path-weighted metrics
    '''
    # stats
    num_successes, num_failures = len(successes), len(failures)
    num_evals = len(successes) + len(failures)
    total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                            sum([entry['path_len_weight'] for entry in failures])
    completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                               sum([entry['completed_goal_conditions'] for entry in failures])
    total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                           sum([entry['total_goal_conditions'] for entry in failures])

    # metrics
    sr = float(num_successes) / num_evals
    pc = completed_goal_conditions / float(total_goal_conditions)
    plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
              total_path_len_weight)
    plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
              total_path_len_weight)

    # result table
    res = dict()
    res['success'] = {'num_successes': num_successes,
                      'num_evals': num_evals,
                      'success_rate': sr}
    res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                    'total_goal_conditions': total_goal_conditions,
                                    'goal_condition_success_rate': pc}
    res['path_length_weighted_success_rate'] = plw_sr
    res['path_length_weighted_goal_condition_success_rate'] = plw_pc

    return res


def compute_alfred_metrics(results : AlfredResults, rollout : List[Dict]):
    task : AlfredTask = rollout[0]["task"]
    traj_data: TrajData = task.traj_data
    r_idx: int = task.repeat_idx

    # trajectory length:
    t = len(rollout) - 1

    # reward:
    total_return = rollout[-1]["return"]

    # check if goal was satisfied
    goal_satisfied = rollout[-1]['md']['goal_satisfied']
    #goal_satisfied = env.get_goal_satisfied()
    success = goal_satisfied

    # goal_conditions
    pcs = rollout[-1]['md']['goal_conditions_met']
    #pcs = env.get_goal_conditions_met()
    goal_condition_success_rate = pcs[0] / float(pcs[1])

    # instruction
    #goal_instr = rollout[-1]['goal_instr']
    goal_instr = str(task)

    # SPL
    path_len_weight = len(traj_data.get_low_actions())
    s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
    pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

    # path length weighted SPL
    plw_s_spl = s_spl * path_len_weight
    plw_pc_spl = pc_spl * path_len_weight

    # log success/fails
    log_entry = {'trial': traj_data.get_task_id(),
                 'type': traj_data.get_task_type(),
                 'repeat_idx': int(r_idx),
                 'goal_instr': goal_instr,
                 'completed_goal_conditions': int(pcs[0]),
                 'total_goal_conditions': int(pcs[1]),
                 'goal_condition_success': float(goal_condition_success_rate),
                 'success_spl': float(s_spl),
                 'path_len_weighted_success_spl': float(plw_s_spl),
                 'goal_condition_spl': float(pc_spl),
                 'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                 'path_len_weight': int(path_len_weight),
                 'reward': float(total_return)}
    if success:
        results.successes.append(log_entry)
    else:
        results.failures.append(log_entry)

    # overall results
    results.results['all'] = get_metrics(results.successes, results.failures)

    print("-------------")
    print("SR: %d/%d = %.3f" % (results.results['all']['success']['num_successes'],
                                results.results['all']['success']['num_evals'],
                                results.results['all']['success']['success_rate']))
    print("GC: %d/%d = %.3f" % (results.results['all']['goal_condition_success']['completed_goal_conditions'],
                                results.results['all']['goal_condition_success']['total_goal_conditions'],
                                results.results['all']['goal_condition_success']['goal_condition_success_rate']))
    print("PLW SR: %.3f" % (results.results['all']['path_length_weighted_success_rate']))
    print("PLW GC: %.3f" % (results.results['all']['path_length_weighted_goal_condition_success_rate']))
    print("-------------")

    # task type specific results
    task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                  'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                  'pick_and_place_with_movable_recep']
    for task_type in task_types:
        task_successes = [s for s in (list(results.successes)) if s['type'] == task_type]
        task_failures = [f for f in (list(results.failures)) if f['type'] == task_type]
        if len(task_successes) > 0 or len(task_failures) > 0:
            results.results[task_type] = get_metrics(task_successes, task_failures)
        else:
            results.results[task_type] = {}

    return results.results

def get_multiple_rollout_metrics_alfred(rollouts):
    alfred_results = AlfredResults()
    for rollout in rollouts:
        if not rollout[0]["task"].is_test():
            compute_alfred_metrics(alfred_results, rollout)
    return alfred_results