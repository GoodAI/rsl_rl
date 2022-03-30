import torch
import torch.nn as nn
from .actor_critic import ActorCritic

class ActorCriticEnvEncoder(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        num_env_params=3,
                        **kwargs):
        if kwargs:
            print("ActorCriticEnvEncoder.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=num_actor_obs - num_env_params + 32,
                         num_critic_obs=num_critic_obs - num_env_params + 32,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        # encoder network
        self.num_env_params = num_env_params
        self.encoder_actor = EnvEncoder(num_env_params)
        self.encoder_critic = EnvEncoder(num_env_params)
        self.extrinsics = None

    def get_extrinsics(self):
        return self.extrinsics

    def act(self, observations, **kwargs):
        env_params = observations[:, -self.num_env_params:]
        extrinsics = self.encoder_actor(env_params)
        self.extrinsics = extrinsics
        new_observations = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1)
        return super().act(new_observations, **kwargs)

    def act_inference(self, observations):
        env_params = observations[:, -self.num_env_params:]
        extrinsics = self.encoder_actor(env_params)
        self.extrinsics = extrinsics
        new_observations = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1)
        return super().act(new_observations)

    def evaluate(self, observations, **kwargs):
        env_params = observations[:, -self.num_env_params:]
        extrinsics = self.encoder_critic(env_params)
        new_observations = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1)
        return super().evaluate(new_observations, **kwargs)


class EnvEncoder(nn.Module):
    def __init__(self, num_env_params) -> None:
        super().__init__()

        layers = []
        # layers.append(nn.Linear(num_env_params, num_env_params * 64))
        layers.append(nn.Linear(num_env_params, 128))
        # layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(num_env_params * 64, num_env_params * 32))
        layers.append(nn.Linear(128, 128))
        # layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(num_env_params * 32, 32))
        layers.append(nn.Linear(128, 32))

        self.net = nn.Sequential(*layers)

    def forward(self, env_params):
        z = self.net(env_params)
        # print(env_params[0], " --> ", z[0])
        return z
