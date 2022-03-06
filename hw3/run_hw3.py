import os
import time

import sys
print(sys.path)


from ift6163.agents.dyna_agent import MBAgent
from ift6163.agents.pg_agent import PGAgent
from ift6163.agents.ac_agent import ACAgent
from ift6163.infrastructure.rl_trainer import RL_Trainer
import hydra, json
from omegaconf import DictConfig, OmegaConf

class pg_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        

        agent_params = {**params['computation_graph_args'],
                         **params['estimate_advantage_args'] ,
                          **params['train_args']}

        tmp = OmegaConf.create({'agent_params' : agent_params })

        self.params = OmegaConf.merge(tmp , params)
        self.params['batch_size_initial'] = self.params['batch_size']

        if self.params['rl_alg'] == 'reinforce':
            agent = PGAgent
        elif self.params['rl_alg'] == 'ac':
            agent = ACAgent
        elif self.params['rl_alg'] == 'dyna':
            agent = MBAgent
        else:
            print("Pick a rl_alg first")
            sys.exit()
        print(self.params)

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params , agent_class =  agent)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


@hydra.main(config_path="conf", config_name="config")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    # print ("params: ", json.dumps(params, indent=4))
    if cfg['env_name']=='reacher-ift6163-v0':
        cfg['ep_len']=200
    if cfg['env_name']=='cheetah-ift6163-v0':
        cfg['ep_len']=500
    if cfg['env_name']=='obstacles-ift6163-v0':
        cfg['ep_len']=100
    params = vars(cfg)
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw3_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + cfg.exp_name + '_' + cfg.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.logdir = logdir

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = pg_Trainer(cfg)
    trainer.run_training_loop()



if __name__ == "__main__":
    import os
    print("Command Dir:", os.getcwd())
    my_main()
