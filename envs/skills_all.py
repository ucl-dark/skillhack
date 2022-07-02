from gym.envs import registration

from nle import nethack
from nle.nethack import Command

from minihack import RewardManager
from envs.mini_skill_transfer import MiniHackSkillTransfer
from reward_manager import AlwaysEvent

MOVE_ACTIONS = tuple(nethack.CompassDirection)

COMMANDS = tuple(
    [
        *MOVE_ACTIONS,
        nethack.Command.PICKUP,
        nethack.Command.PUTON,
        nethack.Command.ZAP,
        nethack.Command.TAKEOFF,
        nethack.Command.WEAR,
        nethack.Command.THROW,
        nethack.Command.ESC,
        nethack.Command.EAT,
        nethack.Command.APPLY,
        nethack.Command.WIELD,
        nethack.Command.QUAFF,  # Only for prep
        ord("$"),
        ord("f"),
        ord("g"),
        ord("h"),
    ]
)

WAND_PREFIXES = [
    "glass",
    "balsa",
    "crystal",
    "maple",
    "pine",
    "oak",
    "ebony",
    "marble",
    "tin",
    "brass",
    "copper",
    "silver",
    "platinum",
    "iridium",
    "zinc",
    "aluminum",
    "uranium",
    "iron",
    "steel",
    "hexagonal",
    "short",
    "runed",
    "long",
    "curved",
    "forked",
    "spiked",
    "jeweled",
]

AMULET_PREFIXES = [
    "circular",
    "spherical",
    "oval",
    "triangular",
    "pyramidal",
    "square",
    "concave",
    "hexagonal",
    "octagonal",
]

RING_PREFIXES = [
    "pearl",
    "iron",
    "twisted",
    "steel",
    "wire",
    "engagement",
    "shiny",
    "bronze",
    "brass",
    "copper",
    "silver",
    "gold",
    "wooden",
    "granite",
    "opal",
    "clay",
    "coral",
    "black onyx",
    "moonstone",
    "tiger eye",
    "jade",
    "agate",
    "topaz",
    "sapphire",
    "ruby",
    "diamond",
    "ivory",
    "emerald",
]


def a_or_an(adj):
    if adj == "uranium":  # Phonetically starts with a constenant
        return "a"
    if adj[0] in ["a", "e", "i", "o", "u"]:
        return "an"
    return "a"


WAND_NAMES = [
    ("- " + a_or_an(pref) + " " + pref + " wand") for pref in WAND_PREFIXES
]
AMULET_NAMES = [
    ("- " + a_or_an(pref) + " " + pref + " amulet") for pref in AMULET_PREFIXES
]
RING_NAMES = [
    ("- " + a_or_an(pref) + " " + pref + " ring") for pref in RING_PREFIXES
]

PENALTY_PER_STEP = 0  # Super hacky and I hate it but quick fix


class MiniHackSkillApplyFrostHorn(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup, so we start with the wand in inventory
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_apply_frost_horn.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(["The lava cools and solidifies."])
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSkillEat(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_eat.des"

        reward_manager = RewardManager()
        reward_manager.add_eat_event("apple")
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSkillFight(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_fight.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            [
                "You kill the wumpus!",
                "You hit the wumpus!",
                "You miss the wumpus.",
            ],
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSkillNavigateBlind(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_navigate_blind.des"

        reward_manager = RewardManager()
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )
        reward_manager.add_message_event(
            ["fdshfdsbyufewj"], reward=0, terminal_required=True
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )

    def step(self, action: int):
        # Drink potion of blindness
        assert "inv_letters" in self._observation_keys

        inv_letters_index = self._observation_keys.index("inv_letters")

        inv_letters = self.last_observation[inv_letters_index]

        if ord("f") in inv_letters:
            self.env.step(Command.QUAFF)
            self.env.step(ord("f"))

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _reward_fn(self, last_observation, action, observation, end_status):
        reward = super()._reward_fn(
            last_observation, action, observation, end_status
        )

        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward += self.reward_win

        return reward


class MiniHackSkillNavigateBlindFixed(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_navigate_blind_fixed.des"

        reward_manager = RewardManager()
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )
        reward_manager.add_message_event(
            ["fdshfdsbyufewj"], reward=0, terminal_required=True
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )

        self.first_step = True

    def step(self, action: int):
        if self.first_step:
            # Drink potion of blindness
            self.first_step = False
            self.env.step(Command.PICKUP)
            self.env.step(Command.QUAFF)
            self.env.step(ord("f"))

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _reward_fn(self, last_observation, action, observation, end_status):
        reward = super()._reward_fn(
            last_observation, action, observation, end_status
        )

        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward += self.reward_win

        return reward


class MiniHackSkillNavigateLava(MiniHackSkillTransfer):
    """Navigate past random lava patches to the staircase"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )
        reward_manager.add_message_event(
            ["fdshfdsbyufewj"], reward=0, terminal_required=True
        )

        super().__init__(
            *args,
            des_file="skills/skill_navigate_lava.des",
            reward_manager=reward_manager,
            **kwargs,
        )

    def _reward_fn(self, last_observation, action, observation, end_status):
        reward = super()._reward_fn(
            last_observation, action, observation, end_status
        )

        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward += self.reward_win

        return reward


class MiniHackSkillNavigateLavaToAmulet(MiniHackSkillTransfer):
    """Navigate past random lava patches to the staircase"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["amulet"],
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_navigate_lava_to_amulet.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillNavigateOverLava(MiniHackSkillTransfer):
    """Navigate over random lava patches to the staircase"""

    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )
        reward_manager.add_message_event(
            ["fdshfdsbyufewj"], reward=0, terminal_required=True
        )

        super().__init__(
            *args,
            des_file="skills/skill_navigate_over_lava.des",
            reward_manager=reward_manager,
            **kwargs,
        )

    def step(self, action: int):
        # Drink potion of levitation
        assert "inv_letters" in self._observation_keys

        inv_letters_index = self._observation_keys.index("inv_letters")

        inv_letters = self.last_observation[inv_letters_index]

        if ord("f") in inv_letters:
            self.env.step(Command.QUAFF)
            self.env.step(ord("f"))

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _reward_fn(self, last_observation, action, observation, end_status):
        reward = super()._reward_fn(
            last_observation, action, observation, end_status
        )

        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward += self.reward_win

        return reward


class MiniHackSkillNavigateWater(MiniHackSkillTransfer):
    """Navigate past random water patches to the staircase"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        # A very hacky reward
        # -0.1 for moving into the water
        # +1 for reaching the staircase
        # But we don't want termination to ever come from the reward manager
        # So we add a dud reward which can never be completed
        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["You try to crawl"],
            reward=-0.1,
            terminal_sufficient=False,
            repeatable=True,
        )
        reward_manager.add_message_event(
            ["fdshfdsbyufewj"], reward=0, terminal_required=True
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_navigate_water.des",
            reward_manager=reward_manager,
            **kwargs,
        )

    def _reward_fn(self, last_observation, action, observation, end_status):
        reward = super()._reward_fn(
            last_observation, action, observation, end_status
        )

        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward += self.reward_win

        return reward


class MiniHackSkillPickUp(MiniHackSkillTransfer):
    """PickUp Item"""

    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_pick_up.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            [
                "f - a silver saber",
                "f - a leather cloak",
                *RING_NAMES,
                "f - a key",
                *WAND_NAMES,
                "f - a dagger",
                "f - a horn",
                "f - a towel",
                "f - a green dragon scale mail",
                "$ - a gold piece",
                *AMULET_NAMES,
            ]
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSkillPutOn(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            [
                "- a ring of levitation",
                "You are now wearing a towel around your head.",
            ]
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_put_on.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillTakeOff(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["You finish taking off your suit."], terminal_sufficient=True
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_take_off.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillThrow(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            [
                "You kill the large mimic",
                "You kill the minotaur",
                "The dagger misses the minotaur",
                "The dagger hits the minotaur",
            ],
            terminal_sufficient=True,
        )

        reward_manager.add_message_event(
            ["Wait!  That's a large mimic!"],
            terminal_required=False,
            reward=-1.5,
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_throw.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillUnlock(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["This door is locked"], terminal_required=False, reward=0.2
        )
        reward_manager.add_message_event(
            ["Unlock it?"], terminal_required=False, reward=0.4
        )
        reward_manager.add_message_event(
            ["You succeed in unlocking the door"], terminal_sufficient=True
        )
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_unlock.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillWear(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(["You are now wearing a robe"])
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_wear.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillWield(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        reward_manager = RewardManager()
        reward_manager.add_message_event(["- a silver saber"])
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args,
            des_file="skills/skill_wield.des",
            reward_manager=reward_manager,
            **kwargs,
        )


class MiniHackSkillZapColdWand(MiniHackSkillTransfer):
    """Zap a wand of cold and put out some lava"""

    def __init__(self, *args, **kwargs):
        # Enable autopickup, so we start with the wand in inventory
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_zap_cold.des"

        reward_manager = RewardManager()
        reward_manager.add_message_event(["The lava cools and solidifies."])
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackSkillZapDeathWand(MiniHackSkillTransfer):
    def __init__(self, *args, **kwargs):
        # Enable autopickup, so we start with the wand in inventory
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("autopickup")
        # Limit Action Space
        kwargs["actions"] = kwargs.pop("actions", COMMANDS)

        des_file = "skills/skill_zap_wod.des"

        reward_manager = RewardManager()
        reward_manager.add_kill_event("minotaur")
        reward_manager.add_event(
            AlwaysEvent(-PENALTY_PER_STEP, True, False, False)
        )

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


registration.register(
    id="MiniHack-Skill-ApplyFrostHorn-v0",
    entry_point="envs.skills_all:" "MiniHackSkillApplyFrostHorn",
)

registration.register(
    id="MiniHack-Skill-Eat-v0",
    entry_point="envs.skills_all:" "MiniHackSkillEat",
)

registration.register(
    id="MiniHack-Skill-Fight-v0",
    entry_point="envs.skills_all:" "MiniHackSkillFight",
)

registration.register(
    id="MiniHack-Skill-NavigateBlind-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateBlind",
)

registration.register(
    id="MiniHack-Skill-NavigateBlindFixed-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateBlindFixed",
)

registration.register(
    id="MiniHack-Skill-NavigateLava-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateLava",
)

registration.register(
    id="MiniHack-Skill-NavigateLavaToAmulet-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateLavaToAmulet",
)

registration.register(
    id="MiniHack-Skill-NavigateOverLava-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateOverLava",
)

registration.register(
    id="MiniHack-Skill-NavigateWater-v0",
    entry_point="envs.skills_all:" "MiniHackSkillNavigateWater",
)

registration.register(
    id="MiniHack-Skill-PickUp-v0",
    entry_point="envs.skills_all:" "MiniHackSkillPickUp",
)

registration.register(
    id="MiniHack-Skill-PutOn-v0",
    entry_point="envs.skills_all:" "MiniHackSkillPutOn",
)

registration.register(
    id="MiniHack-Skill-TakeOff-v0",
    entry_point="envs.skills_all:" "MiniHackSkillTakeOff",
)

registration.register(
    id="MiniHack-Skill-Throw-v0",
    entry_point="envs.skills_all:" "MiniHackSkillThrow",
)

registration.register(
    id="MiniHack-Skill-Unlock-v0",
    entry_point="envs.skills_all:" "MiniHackSkillUnlock",
)

registration.register(
    id="MiniHack-Skill-Wear-v0",
    entry_point="envs.skills_all:" "MiniHackSkillWear",
)

registration.register(
    id="MiniHack-Skill-Wield-v0",
    entry_point="envs.skills_all:" "MiniHackSkillWield",
)

registration.register(
    id="MiniHack-Skill-ZapColdWand-v0",
    entry_point="envs.skills_all:" "MiniHackSkillZapColdWand",
)

registration.register(
    id="MiniHack-Skill-ZapWoD-v0",
    entry_point="envs.skills_all:" "MiniHackSkillZapDeathWand",
)

registration.register(
    id="MiniHack-Skill-ZapWoD-v1",
    entry_point="envs.skills_all:" "MiniHackSkillZapDeathWand",
)
