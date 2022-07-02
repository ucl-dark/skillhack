import torch

from agent.polybeast.models import BaseNet

from agent.polybeast.skill_transfer.skill_transfer import load_model


class FOCNet(BaseNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        options_path = flags.foc_options_path
        configs_path = flags.foc_options_config_path

        if len(options_path) != len(configs_path):
            print(options_path)
            print(configs_path)
            raise ValueError(
                "options_path length does not equal configs_length "
                + str(len(options_path))
                + " "
                + str(len(configs_path))
            )

        self.num_options = len(options_path)

        super(FOCNet, self).__init__(
            observation_shape, self.num_options, flags, device
        )

        self.options = [
            load_model(flags.env, paths[0], paths[1], device)[0]
            for paths in zip(options_path, configs_path)
        ]

    def forward(self, inputs, core_state, learning=False):
        (output, core_state) = super().forward(inputs, core_state, learning)

        with torch.no_grad():
            option_outs = [
                self.options[i](inputs, core_state, learning)
                for i in range(len(self.options))
            ]

        batch_size = output["policy_logits"].shape[0]
        num_actors = output["policy_logits"].shape[1]

        action = torch.zeros((batch_size, num_actors), dtype=torch.int64)

        for i in range(batch_size):
            for j in range(num_actors):
                ind = output["action"][i][j]

                # ind = input("> ")
                # if not ind:
                #    ind = 0
                # else:
                #    ind = int(ind)

                # Select action according to Policy Over Options
                action[i, j] = option_outs[ind][0]["action"][i][j]

        return (
            dict(
                policy_logits=output["policy_logits"],
                baseline=output["baseline"],
                action=action,
                chosen_option=output["action"],
                teacher_logits=output["policy_logits"],
                pot_sm=output["policy_logits"],
            ),
            core_state,
        )
