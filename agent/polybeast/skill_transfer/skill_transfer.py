from omegaconf import OmegaConf
import torch

from nle.env.base import DUNGEON_SHAPE

from agent.common.envs import tasks
from agent.polybeast.models import BaseNet


def create_basenet(flags, device):
    model_string = flags.model
    if model_string == "baseline":
        model_cls = BaseNet
    elif model_string == "cnn" or model_string == "transformer":
        raise RuntimeError(
            "model=%s deprecated, use model=baseline crop_model=%s instead"
            % (model_string, model_string)
        )
    else:
        raise NotImplementedError("model=%s" % model_string)

    num_actions = len(
        tasks.ENVS[flags.env](savedir=None, archivefile=None)._actions
    )

    model = model_cls(DUNGEON_SHAPE, num_actions, flags, device)
    model.to(device=device)
    return model


def load_model(env, pretrained_path, pretrained_config_path, device):
    flags = OmegaConf.load(pretrained_config_path)
    flags["env"] = env
    model = create_basenet(flags, device)

    checkpoint_states = torch.load(pretrained_path, map_location=device)

    model.load_state_dict(checkpoint_states["model_state_dict"])
    # model.training = False

    hidden = model.initial_state(batch_size=1)
    return model, hidden
