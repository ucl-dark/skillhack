from gym.envs import registration

from minihack import RewardManager

from envs import skills_all
from envs.mini_skill_transfer import MiniHackSkillTransfer
from reward_manager import IntersectionRewardManager


class MiniHackSimpleSeq(MiniHackSkillTransfer):
    """Simple Sequence of skills"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        des_file = "tasks/task_simple_seq.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["You kill the wumpus!"],
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSimpleIntersection(MiniHackSkillTransfer):
    """Simple Intersection of skills"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        des_file = "tasks/task_simple_intersection.des"

        reward_manager = IntersectionRewardManager()
        reward_manager.add_eat_event("apple", reward=0.5)
        reward_manager.add_wear_event("leather cloak", reward=0.5)

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSimpleRandom(MiniHackSkillTransfer):
    """Simple Randomised sequence of skills"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        des_file = "tasks/task_simple_random.des"

        reward_manager = RewardManager()
        reward_manager.add_kill_event("minotaur", terminal_sufficient=True)
        reward_manager.add_message_event(
            ["The dagger misses the minotaur", "The dagger hits the minotaur"],
            terminal_sufficient=True,
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSimpleUnion(MiniHackSkillTransfer):
    """Simple Union of skills"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", skills_all.COMMANDS)

        des_file = "tasks/task_simple_union.des"

        super().__init__(*args, des_file=des_file, **kwargs)


registration.register(
    id="MiniHack-SimpleSeq-v0",
    entry_point="envs.task_simple:" "MiniHackSimpleSeq",
)

registration.register(
    id="MiniHack-SimpleIntersection-v0",
    entry_point="envs.task_simple:" "MiniHackSimpleIntersection",
)

registration.register(
    id="MiniHack-SimpleRandom-v0",
    entry_point="envs.task_simple:" "MiniHackSimpleRandom",
)

registration.register(
    id="MiniHack-SimpleUnion-v0",
    entry_point="envs.task_simple:" "MiniHackSimpleUnion",
)
