import torch

from agent.polybeast.models.base import BaseNet
from agent.polybeast.skill_transfer.skill_transfer import load_model


class KSNet(BaseNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(KSNet, self).__init__(
            observation_shape, num_actions, flags, device
        )

        teacher_paths = flags.teacher_path
        config_paths = flags.teacher_config_path

        self.num_teachers = len(teacher_paths)

        self.teachers = [
            load_model(flags.env, teacher_paths[i], config_paths[i], device)[0]
            for i in range(self.num_teachers)
        ]

    def forward(self, inputs, core_state, learning=False):
        (output, core_state) = super().forward(inputs, core_state, learning)

        with torch.no_grad():
            (teacher_output_avg, _) = self.teachers[0].forward(
                inputs, core_state, learning
            )
            teacher_output_avg = teacher_output_avg["policy_logits"]
            teacher_output_avg = torch.softmax(teacher_output_avg, 2)

            for i in range(1, self.num_teachers):
                (teacher_output, _) = self.teachers[i].forward(
                    inputs, core_state, learning
                )
                teacher_output_avg += torch.softmax(
                    teacher_output["policy_logits"], 2
                )

            teacher_output_avg /= self.num_teachers

        return (
            dict(
                policy_logits=output["policy_logits"],
                baseline=output["baseline"],
                action=output["action"],
                chosen_option=output["action"],
                teacher_logits=teacher_output_avg,  # TODO not actually logits anymore
                pot_sm=output["policy_logits"],
            ),
            core_state,
        )
