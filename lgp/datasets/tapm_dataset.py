from typing import List, Dict, Union
import itertools
import copy
import torch

from lgp.abcd.model_factory import ModelFactory
from lgp.abcd.dataset import ExtensibleDataset
from lgp.rollout.rollout_data import load_rollout_from_path

from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr

MAX_LEN = 20
DROP_EXPLORE = True


class TapmDataset(ExtensibleDataset):
    """
    Dataset that loads a list of rollouts, and returns (task, state, action, reward, nextstate, nextstatevalue) batches.

    A rollout is a sequence of (observation, action, reward, done).
    An "Observation Function" is used to integrate observations over time into state representations.
    To provide value targets, the rewards are summed up in inverse chronological order across the rollout
    """

    def __init__(self,
                 rollout_paths: List[str],
                 dataset_mode : str,
                 model_factory: ModelFactory,
                 gamma : float,
                 listbatch : float = False):
        self.rollout_paths = rollout_paths
        self.gamma = gamma

        self.obsfunc = model_factory.get_observation_function()
        #self.action_repr_function = model_factory.get_action_repr_function()
        self.task_repr_function = model_factory.get_task_repr_function()

        self.listbatch = listbatch

    def __len__(self):
        return len(self.rollout_paths)

    def __getitem__(self, i):
        rollout = load_rollout_from_path(self.rollout_paths[i])
        rollout = rollout[:MAX_LEN] # Clip to max length of 10 for now
        if len(rollout) == 0:
            print("SKIPPING EMPTY ROLLOUT")
            return self.__getitem__(i+1)
        if self._is_rollout_processed(rollout):
            return rollout
        else:
            return self._compress_rollout(self._preprocess_rollout(rollout))

    def _is_rollout_processed(self, rollout):
        return "state_preproc" in rollout[0]

    def _preprocess_rollouts(self, rollouts: List[List[Dict]]):
        rollouts_o = []
        for i, rollout in enumerate(rollouts):
            rollout_o = self._preprocess_rollout(rollout, i)
            rollouts_o.append(rollout_o)
        return rollouts_o

    def _preprocess_rollout(self, rollout: List[Dict], rollout_tag = None):
        # Compute value targets
        value = 0
        for sample in reversed(rollout):
            sample["next_value"] = value
            value = value * self.gamma + sample["reward"]
            sample["value"] = value

        # Fix bug if last action is not stop
        if len(rollout) > 0 and (not rollout[-1]["subgoal"].is_stop()):
            rollout[-1] = copy.deepcopy(rollout[-1])
            rollout[-1]["subgoal"] = AlfredSubgoal.from_type_str_and_arg_id("Stop", -1)

        # Drop explore actions
        if DROP_EXPLORE:
            rollout_out = []
            drop_now = False
            for i, sample in enumerate(rollout):
                if drop_now:
                    prev_sample = rollout[i-1]
                    sample_out = {
                        "task": sample["task"],
                        "state_repr": prev_sample["state_repr"],
                        "observation": prev_sample["observation"],
                        "action": sample["action"],
                        "action_repr": sample["action_repr"],
                        "reward": sample["reward"],
                        "done": sample["done"],
                        "remark": sample["remark"],
                        "subgoal": sample["subgoal"],
                        "eventual_action_ll": sample["eventual_action_ll"],
                        "eventual_action_observation": sample["eventual_action_observation"],
                        "next_value": sample["next_value"],
                        "value": sample["value"]
                    }
                    rollout_out.append(sample_out)
                    drop_now = False
                elif sample["subgoal"].type_str() == "Explore":
                    drop_now = True
                else:
                    rollout_out.append(sample)
            rollout = rollout_out

        # Compute state representations by integrating observations
        state_repr = None
        for sample in rollout:
            # If state representation was NOT precomputed, compute it
            if "state_repr" not in sample:
                state_repr = self.obsfunc(sample["observation"], state_repr)    # This has memory
                sample["state_repr"] = state_repr

            #if "task_repr" not in sample:
            #    task_repr = self.task_repr_function(sample["task"])
            #    sample["task_repr"] = task_repr

            # Currently the last action is None
            #if "action_repr" not in sample and sample["action"] is not None:
            #    action_repr = self.action_repr_function(sample["action"], sample["observation"])
            #    sample["action_repr"] = action_repr

            #if "action_repr_hl" not in sample and sample["subgoal"] is not None:
            #    action_repr_hl = self.action_repr_function(sample["subgoal"], None)
            #    sample["action_repr_hl"] = action_repr_hl

        # Add "next_state_repr" to each sample
        next_state_repr = None
        for sample in reversed(rollout):
            sample["next_state_repr"] = next_state_repr
            next_state_repr = sample["state_repr"]

        # Mark each sample with the rollout index (used to identify "validation" examples for Platt Scaling)
        for sample in rollout:
            sample["rollout_index"] = rollout_tag

        return rollout

    def _compress_rollout(self, rollout):
        rollout_out = []
        from lgp.models.alfred.hlsm.transformer_modules.state_repr_encoder_pooled import StateReprEncoderPooled
        for sample in rollout:
            sample_out = {
                "task": sample["task"],
                #"action_repr_hl": sample["action_repr_hl"],
                "subgoal": sample["subgoal"],
                "rollout_index": sample["rollout_index"],
                "eventual_action_ll": sample["eventual_action_ll"],
                "eventual_action_observation": sample["eventual_action_observation"],
                "state_preproc": StateReprEncoderPooled._make_pooled_repr(sample["state_repr"]),
                "state_repr": sample["state_repr"]
            }
            rollout_out.append(sample_out)
        return rollout_out

    @classmethod
    def extract_touch_argument(cls, action, observation):
        # B x C x W x L x H
        # state_repr = batch["states"][t]
        if action.argument_mask is None:
            return -1  # TODO: Check what to do here actually.
        semantic_image = observation.semantic_image[0]
        masked_semantics = action.argument_mask[None, :, :].to(semantic_image.device) * semantic_image
        semantic_vector = masked_semantics.sum(1).sum(1)
        argclass = semantic_vector.argmax().item()
        return argclass

    def _calc_batch_id_list(self, rollout_indices):
        batch_id = []
        pr = -1
        b = -1
        for r in rollout_indices:
            if r != pr:
                b += 1
            pr = r
            batch_id.append(b)
        return batch_id

    def _collate_one(self, list_of_examples: List[Dict]):
        print(len(list_of_examples))

        # Sometimes we sample multiple very long rollouts and otherwise result in out-of-memory
        if len(list_of_examples) > MAX_LEN:
            print(f"Pruning example from: {len(list_of_examples)} to {MAX_LEN}")
            list_of_examples = list_of_examples[:MAX_LEN]

        tasks = [l["task"] for l in list_of_examples]
        task_reprs = self.task_repr_function(tasks)
        states_preproc = [l["state_preproc"] for l in list_of_examples]
        states = [l["state_repr"] for l in list_of_examples]
        #action_reprs_hl = [l["action_repr_hl"] if "action_repr_hl" in l else l["action_repr_NOPE"] for l in
        #                   list_of_examples]
        subgoals = [l["subgoal"] for l in list_of_examples]
        rollout_indices = [l["rollout_index"] for l in list_of_examples]

        # Build a list indicating which batch element each sample belongs to.
        batch_id = self._calc_batch_id_list(rollout_indices)

        # Construct high-level actions
        # Action types from high-level
        # Action arguments from eventual low-level
        #eventual_actions = [l["eventual_action_ll"] for l in list_of_examples]
        #eventual_observations = [l["eventual_action_observation"] for l in list_of_examples]

        #raise NotImplementedError("Figure out which of the two subgoals are needed and applicable")

        # Semantic actions
        #semantic_action_args = [self.extract_touch_argument(a, o) for a, o in
        #                        zip(eventual_actions, eventual_observations)]
        #subgoals = []
        #for arg, act in zip(semantic_action_args, subgoals):
        #    subgoal = AlfredSubgoal.from_type_str_arg_id_with_mask(act.action_type, arg, act.argument_mask.data)
        #    subgoals.append(subgoal)
        subgoals = AlfredSubgoal.collate(subgoals)

        states_preproc = torch.cat(states_preproc, dim=0)
        states = AlfredSpatialStateRepr.collate(states)

        batch = {
            "task_reprs": task_reprs,
            "states": states,
            "states_preproc": states_preproc,
            "subgoals": subgoals,
            "batch_id": batch_id
        }
        return batch

    # Inherited from lgp.abcd.dataset.ExtensibleDataset
    def collate_fn(self, list_of_examples: Union[List[Dict], List[List[Dict]]]) -> Dict:
        collate_one_fn = self._collate_one
        if self.listbatch:
            out = [collate_one_fn(l) for l in list_of_examples]
            return out
        else:
            # If the list of examples is actually a list of rollouts, chain the rollouts
            if isinstance(list_of_examples[0], List):
                list_of_examples = list(itertools.chain(*list_of_examples))
            return collate_one_fn(list_of_examples)

