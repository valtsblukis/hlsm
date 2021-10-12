from typing import Dict, Union
from lgp.abcd.model_factory import ModelFactory
from lgp.abcd.agent import Agent
from lgp.abcd.skill import Skill

# High-level reasoning models
from lgp.models.alfred.hlsm.hlsm_observation_function import HlsmObservationFunction
from lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskReprFunction
from lgp.models.alfred.hlsm.hlsm_subgoal_model import HlsmSubgoalModel

# Low-level skills
from lgp.models.alfred.handcoded_skills.go_for import GoForSkill
from lgp.models.alfred.handcoded_skills.go_for_manual import GoForManualSkill
from lgp.models.alfred.handcoded_skills.explore_skill import ExploreSkill
from lgp.models.alfred.handcoded_skills.interact_skill import InteractSkill
from lgp.models.alfred.handcoded_skills.init_skill import InitSkill

from lgp.parameters import Hyperparams


class HlsmModelFactory(ModelFactory):
    def __init__(self, hparams: Hyperparams):
        super().__init__()
        self.hparams = hparams

        self.use_explore_skill = hparams.get("use_explore_skill", False)
        self.use_heuristic_gofor_skill = hparams.get("use_heuristic_gofor_skill", False)
        print(f"USE HEURISTIC GOFOR SKILL: {self.use_heuristic_gofor_skill}")

    def get_skillset(self) -> Dict[str, Skill]:
        # Returns the set of skills (low-level agents) used by the hierarchical agent

        # First picks a point on the exploration frontier
        # Then invokes goto skill to go to that point
        if self.use_explore_skill:
            explore_skill = ExploreSkill()
            explore_skill_b = ExploreSkill()
        else:
            print("NOT USING EXPLORATION SKILL!!!")
            explore_skill = None
            explore_skill_b = None

        # First finds a pose from which the action can be executed
        # Then invokes the goto skill to go to this pose
        if self.use_heuristic_gofor_skill:
            gofor_skill = GoForManualSkill()
        else:
            gofor_skill = GoForSkill()

        # First deploys gofor skill to bring the agent to a point
        # from which the action can be executed
        # Then emits the corresponding interaction action in the first-person frame
        interact_skill = InteractSkill(gofor_skill, explore_skill_b)

        init_skill = InitSkill()

        skillset = {
            "OpenObject" : interact_skill,
            "CloseObject" : interact_skill,
            "PickupObject" : interact_skill,
            "PutObject" : interact_skill,
            "ToggleObjectOn" : interact_skill,
            "ToggleObjectOff" : interact_skill,
            "SliceObject" : interact_skill,

            # This does not correspond to any high-level actions:
            "init": init_skill
        }

        if explore_skill is not None:
            skillset["Explore"] = explore_skill

        return skillset

    def get_observation_function(self) -> HlsmObservationFunction:
        return HlsmObservationFunction(self.hparams)

    def get_subgoal_model(self) -> HlsmSubgoalModel:
        return HlsmSubgoalModel(self.hparams)

    def get_task_repr_function(self) -> "HlsmTaskReprFunction":
        return HlsmTaskReprFunction()