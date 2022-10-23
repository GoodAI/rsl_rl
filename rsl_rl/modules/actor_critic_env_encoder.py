import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .actor_critic import ActorCritic

import pandas as pd

class RMADataset(Dataset):
    def __init__(self, df, envs=50, prev_actions=20):
        self.data = df
        self.envs = envs
        self.prev_actions = prev_actions
        
    def __len__(self):
        return len(self.data)

    def get_row_data(self, idx):
        loc = self.data.iloc[idx]
        label = loc["extrinsics"]
        actions = loc["actions"]
        xt = loc["X"]
        orientation = loc["orientation"]
        
        return label, actions, xt, orientation
    
    def __getitem__(self, idx):
        label, actions, xt, orientation = self.get_row_data(idx)
        
        input_len = len(actions) + len(xt) + len(orientation)
        # placeholder prefilled with zeros
        data = [0.0] * self.prev_actions * input_len
        data[0:input_len] = actions + xt + orientation
        start_id = input_len

        for next_idx in range(idx - self.envs, max(0, idx - (self.prev_actions -1) * self.envs), -self.envs):
            _, actions, xt, orientation = self.get_row_data(next_idx)
            data[start_id:start_id + input_len] = actions + xt + orientation
            start_id += input_len
        
        return torch.tensor(label), torch.tensor([data])

    def get_data(self):
        data = []
        for i in range(len(self.data) - self.envs, len(self.data)):
            data.append(self[i][1].unsqueeze(0))

        return torch.cat(data)

class AdaptationModuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv1d(1, 32, 3, stride=3)
        conv2 = nn.Conv1d(32, 64, 5, stride=3)
        conv3 = nn.Conv1d(64, 128, 5, stride=2)

        conv4 = nn.Conv1d(128, 64, 7, stride=2)
        conv5 = nn.Conv1d(64, 64, 5, stride=2)
        conv6 = nn.Conv1d(64, 32, 5, stride=1)
        conv7 = nn.Conv1d(32, 32, 3, stride=1)
        
        self.layers = nn.Sequential(conv1, nn.BatchNorm1d(32), nn.ReLU(),
                                    conv2, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv3, nn.BatchNorm1d(128), nn.ReLU(),
                                    conv4, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv5, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv6, nn.BatchNorm1d(32), nn.ReLU(),
                                    conv7, nn.Flatten())

    def forward(self, x):
        x = self.layers(x)
        return x


class ActorCriticEnvEncoder(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        num_env_params=12,
                        xt_index = -1,
                        action_index = -1,
                        orientation_index=-1,
                        use_adaptation=False,
                        adaptation_path=None, # path to pytorch checkpoint
                        prepare_training_data=False,
                        training_data_path="training_data.hkl.zip",
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
        self.extra_data = pd.DataFrame()
        self.adaptation_module = AdaptationModuleNet() if use_adaptation else None
        self.adaptation_path = adaptation_path
        self.prepare_training_data = prepare_training_data
        self.training_data_path = training_data_path

        self.xt_index = xt_index
        self.action_index = action_index
        self.orientation_index = orientation_index
        self.iter = 0

    def get_extrinsics(self):
        return self.extrinsics

    def save_extra_data(self, path):
        self.extra_data.to_hdf(path)

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

        if self.iter == 0 and self.adaptation_module:
            self.adaptation_module.load_state_dict(torch.load(self.adaptation_path))
            self.adaptation_module.eval()
            self.adaptation_module.to("cuda:0")

        if self.prepare_training_data:
            if self.iter < 1000:
                df = pd.DataFrame()
                df["extrinsics"] = extrinsics.detach().cpu().numpy().tolist()
                df["actions"] = observations[:, self.action_index:self.action_index + self.num_actions].detach().cpu().numpy().tolist()
                df["X"] = observations[:, self.xt_index:self.xt_index + self.num_actions].detach().cpu().numpy().tolist()
                df["orientation"] = observations[:, self.orientation_index:self.orientation_index+3].detach().cpu().numpy().tolist()
                df["observations"] = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1).detach().cpu().numpy().tolist()
                self.extra_data = pd.concat((self.extra_data, df), ignore_index=True)

            if self.iter == 1000:
                self.extra_data.to_pickle(self.training_data_path)
                self.extra_data = pd.DataFrame()
                print("saved")

        if self.adaptation_module:
            df = pd.DataFrame()
            df["extrinsics"] = extrinsics.detach().cpu().numpy().tolist()
            df["actions"] = observations[:, self.action_index:self.action_index + self.num_actions].detach().cpu().numpy().tolist()
            df["X"] = observations[:, self.xt_index:self.xt_index + self.num_actions].detach().cpu().numpy().tolist()
            df["orientation"] = observations[:, self.orientation_index:self.orientation_index+3].detach().cpu().numpy().tolist()
            df["observations"] = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1).detach().cpu().numpy().tolist()
            self.extra_data = pd.concat((self.extra_data, df), ignore_index=True)
            rma_dataset = RMADataset(self.extra_data)
            rma_data = rma_dataset.get_data().to("cuda:0")
            rma_result = self.adaptation_module(rma_data)

        if not self.adaptation_module:
            new_observations = torch.cat((observations[:, :-self.num_env_params], extrinsics), dim=-1)
        else:
            new_observations = torch.cat((observations[:, :-self.num_env_params], rma_result), dim=-1)

        self.iter += 1

        return super().act_inference(new_observations)

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
