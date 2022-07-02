from minihack import MiniHackSkill
import os
from os.path import dirname


class MiniHackSkillTransfer(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            reward_win=1,
            reward_lose=-1,
            **kwargs,
        )

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Use reward_manager to collect reward calculated in _is_episode_end,
        or revert to standard sparse reward."""
        if self.reward_manager is not None:
            reward = self.reward_manager.collect_reward()

            # Also include negative penalty for death
            if (
                end_status == self.StepStatus.DEATH
                or end_status == self.StepStatus.ABORTED
            ):
                reward += self.reward_lose
        else:
            if end_status == self.StepStatus.TASK_SUCCESSFUL:
                reward = self.reward_win
            elif end_status == self.StepStatus.RUNNING:
                reward = 0
            else:  # death or aborted
                reward = self.reward_lose
        return reward + self._get_time_penalty(last_observation, observation)

    # Set des_file to the absolute path
    def _patch_nhdat(self, des_file):
        if des_file.endswith(".des"):
            des_path = os.path.abspath(des_file)
            if not os.path.exists(des_path):
                des_path = os.path.abspath(
                    os.path.join(
                        dirname(dirname(os.path.realpath(__file__))) + "/data",
                        des_file,
                    )
                )
                return super()._patch_nhdat(des_path)

        return super()._patch_nhdat(des_file)
