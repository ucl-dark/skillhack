import torch


from torch import nn
from torch.nn import functional as F

from agent.polybeast.models import BaseNet
from agent.polybeast.models.base import NUM_CHARS
from agent.polybeast.skill_transfer.skill_transfer import load_model


class HKSNet(BaseNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(HKSNet, self).__init__(
            observation_shape, num_actions, flags, device
        )

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

        self.options = [
            load_model(flags.env, paths[0], paths[1], device)[0]
            for paths in zip(options_path, configs_path)
        ]

        # self.pot = BaseNet(observation_shape, self.num_options, flags, device)

        self.pot_layer = nn.Linear(self.h_dim, self.num_options)

    def forward(self, inputs, core_state, learning=False):
        T, B, *_ = inputs["glyphs"].shape

        glyphs, features = self.prepare_input(inputs)

        # -- [B x 2] x,y coordinates
        coordinates = features[:, :2]

        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)
        if self.equalize_input_dim:
            features_emb = self.project_feature_dim(features_emb)

        assert features_emb.shape[0] == T * B

        reps = [features_emb]

        # -- [B x H' x W']
        crop = self.glyph_embedding.GlyphTuple(
            *[self.crop(g, coordinates) for g in glyphs]
        )
        # -- [B x H' x W' x K]
        crop_emb = self.glyph_embedding(crop)

        if self.crop_model == "transformer":
            # -- [B x W' x H' x K]
            crop_rep = self.extract_crop_representation(crop_emb, mask=None)
        elif self.crop_model == "cnn":
            # -- [B x K x W' x H']
            crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
            # -- [B x W' x H' x K]
            crop_rep = self.extract_crop_representation(crop_emb)
        # -- [B x K']

        crop_rep = crop_rep.view(T * B, -1)
        if self.equalize_input_dim:
            crop_rep = self.project_crop_dim(crop_rep)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self.glyph_embedding(glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)
        if self.equalize_input_dim:
            glyphs_rep = self.project_glyph_dim(glyphs_rep)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = inputs["message"].long().view(T * B, -1)
            if self.msg_model == "cnn":
                # convert messages to one-hot, [T * B x 96 x 256]
                one_hot = F.one_hot(messages, num_classes=NUM_CHARS).transpose(
                    1, 2
                )
                char_rep = self.conv2_6_fc(self.conv1(one_hot.float()))
            elif self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
            else:  # lstm, gru
                char_emb = self.char_lt(messages)
                output = self.char_rnn(char_emb)[0]
                fwd_rep = output[:, -1, : self.h_dim // 2]
                bwd_rep = output[:, 0, self.h_dim // 2 :]
                char_rep = torch.cat([fwd_rep, bwd_rep], dim=1)

            if self.equalize_input_dim:
                char_rep = self.project_msg_dim(char_rep)
            reps.append(char_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * t for t in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1
            )
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            chosen_option=action,
            teacher_logits=policy_logits,
            pot_sm=policy_logits,
        )

        #
        #
        #
        #
        #

        # (output, core_state) = super().forward(inputs, core_state, learning)

        pot_logits = self.pot_layer(core_output)
        pot_logits = pot_logits.view(T, B, self.num_options)

        pot_sm = torch.softmax(pot_logits, 2)

        with torch.no_grad():
            option_sm = [
                torch.softmax(
                    self.options[i](inputs, core_state, learning)[0][
                        "policy_logits"
                    ],
                    2,
                )
                for i in range(len(self.options))
            ]

        weighted_teacher = (
            pot_sm[:, :, 0].unsqueeze(2).repeat(1, 1, self.num_actions)
            * option_sm[0]
        )

        for i in range(1, self.num_options):
            weighted_teacher += (
                pot_sm[:, :, i].unsqueeze(2).repeat(1, 1, self.num_actions)
                * option_sm[i]
            )

        return (
            dict(
                policy_logits=output["policy_logits"],
                baseline=output["baseline"],
                action=output["action"],
                chosen_option=output["action"],
                teacher_logits=weighted_teacher,  # TODO not actually logits anymore
                pot_sm=pot_sm,
            ),
            core_state,
        )
