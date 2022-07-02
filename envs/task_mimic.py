from gym.envs import registration

from minihack import RewardManager

from envs import skills_all
from envs.mini_skill_transfer import MiniHackSkillTransfer


class MiniHackMimic(MiniHackSkillTransfer):
    """PickUp a wand in a random location"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        des_file = "tasks/task_mimic.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            [
                "The dagger hits the large mimic",
                "The dagger misses",
                "amulet (being worn)",
            ],
            terminal_sufficient=True,
        )

        reward_manager.add_message_event(
            ["Wait!  That's a large mimic!"],
            terminal_sufficient=True,
            reward=-1,
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


registration.register(
    id="MiniHack-Mimic-v0",
    entry_point="envs.task_mimic:" "MiniHackMimic",
)
