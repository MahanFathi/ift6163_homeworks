import functools
from itertools import accumulate
from collections import OrderedDict
from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *
import random
import numpy as np


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.gamma = self.agent_params['discount']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_env = int(num_data / self.ensemble_size)
        data_indices = np.arange(num_data)

        #########################################
        #           TRAIN DYNA MODELS           #
        #########################################

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            # Copy this from previous homework

            indices = np.random.choice(data_indices, num_data_per_env)
            observations = ob_no[indices]
            actions = ac_na[indices]
            next_observations = next_ob_no[indices]

            model = self.dyn_models[i]
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)
            
        # TODO Pick a model at random
        # TODO Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution)
        # resample trajectories NOTE: We need recent *trajectories* not just random transitions
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy
        # TODO add this generated data to the real data
        # TODO Perform a policy gradient update
        # Hint: Should the critic be trained with this generated data? Try with and without and include your findings in the report.
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.replay_buffer.sample_recent_data(ob_no.shape[0], False)


        horizon = re_batch[0].shape[0]
        for _ in range(len(re_batch)):
            obs, acs, rewards, next_obs, terminals = [], [], [], [], []
            ob = ob_no[random.choice(range(ob_no.shape[0]))][np.newaxis, :]
            model = random.choice(self.dyn_models)
            for i in range(horizon):
                obs.append(ob)
                ac = self.actor.get_action(ob)
                acs.append(ac)
                next_ob = model.get_prediction(ob, ac, self.data_statistics)
                next_obs.append(next_ob)
                terminals.append(0)
                ob = next_ob

            terminals[-1] = 1
            con_obs = np.concatenate(obs, axis=0)
            con_acs = np.concatenate(acs, axis=0)
            con_next_obs = np.concatenate(next_obs, axis=0)
            terminals = np.stack(terminals)

            ob_batch = np.concatenate([ob_batch, con_obs], axis=0)
            ac_batch = np.concatenate([ac_batch, con_acs], axis=0)
            next_ob_batch = np.concatenate([next_ob_batch, con_next_obs], axis=0)
            terminal_batch = np.concatenate([terminal_batch, terminals], axis=0)
            imagined_rewards = self.env.get_reward(con_obs, con_acs)[0]
            re_batch.append(imagined_rewards)


        q_values = self.calculate_q_vals(re_batch)
        advantages = self.estimate_advantage(ob_batch, re_batch, q_values, terminal_batch)
        train_log = self.actor.update(ob_batch, ac_batch, advantages, q_values)

        return train_log
        # loss = OrderedDict()
        # re_batch = np.concatenate(re_batch, axis=0)
        # loss['Critic_Loss'] = self.critic.update(ob_batch, ac_batch, next_ob_batch, re_batch, terminal_batch)
        # adv_batch = self.estimate_advantage(ob_batch, next_ob_batch, re_batch, terminal_batch)
        # loss['Actor_Loss'] = self.actor.update(ob_batch, ac_batch, adv_batch)
        # loss['FD_Loss'] = np.mean(losses)
        # return loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)


    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        discounted_return = functools.reduce(
            lambda ret, reward: ret * self.gamma + reward,
            reversed(rewards),
        )
        list_of_discounted_returns =  [discounted_return] * len(rewards)

        return list_of_discounted_returns


    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        list_of_discounted_cumsums = list(
            accumulate(reversed(rewards), lambda ret, reward: ret * self.gamma + reward))[::-1]

        return list_of_discounted_cumsums


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(r) for r in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            # values = TODO
            values = values_unnormalized*np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula
                    # y=45 ## Remove: This is just to help with compiling
                    if terminals[i]==1:
                        advantages[i] = rews[i] - values[i]
                    else:
                        advantages[i] = rews[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] += self.gamma * self.gae_lambda * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            advantages = (advantages - np.mean(advantages))/(np.std(advantages)+0.0001)

        return advantages
