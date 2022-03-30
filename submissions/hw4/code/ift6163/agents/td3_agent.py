import numpy as np

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.td3_critic import TD3Critic
import copy

from ift6163.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.q_fun = TD3Critic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        # self.replay_buffer_idx = -1
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # TODO add noise to the deterministic policy
        # perform_random_action = TODO
        eps = 0.1 # let's exploit 90% of the times and explore for the rest
        perform_random_action = (np.random.random()<eps) or (self.t<self.learning_starts)
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            action = self.actor.get_action(self.replay_buffer.encode_recent_observation())
            action += np.random.normal(0.0, self.exploration_noise, size=action.shape[0])
            action = np.clip(action, -1.0, 1.0)
        # NOTE: we don't add noise to the action anymore (not the case in TD3)
        # HINT: take random action

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        self.last_obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()


    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.q_fun.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n,
            )

            # TODO fill in the call to the update function using the appropriate tensors
            ## Hint the actor will need a copy of the q_net to maximize the Q-function
            if self.num_param_updates % self.policy_update_frequency:
                self.actor.update(
                    ob_no, self.q_fun,
                )

            # TODO update the target network periodically
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                # TODO
                self.q_fun.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
