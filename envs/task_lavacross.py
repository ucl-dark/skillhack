from gym.envs import registration

from envs import skills_all
from envs.mini_skill_transfer import MiniHackSkillTransfer


class MiniHackLCFreeze(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        super().__init__(
            *args,
            des_file="tasks/task_lavacross_freeze.des",
            **kwargs,
        )


registration.register(
    id="MiniHack-LavaCrossFreeze-v0",
    entry_point="envs.task_lavacross:" "MiniHackLCFreeze",
)
