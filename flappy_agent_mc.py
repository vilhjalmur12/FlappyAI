from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import operator
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

discount = 0.1
learning_rate = 0.1


class FlappyAgent:

    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.epsilon = 0.1
        self.gamma = 0.1

        self.environment = {}

        player_vel, player_y, next_pipe_top_y, next_pipe_dist_to_player = self.environment_variables()
        self.environment['player_vel'] = player_vel
        self.environment['player_y'] = player_y
        self.environment['next_pipe_top_y'] = next_pipe_top_y
        self.environment['next_pipe_dist_to_player'] = next_pipe_dist_to_player

        self.Q = self._init_Q()


    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}


    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return


    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #print("state: %s" % str(state))
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        best_action = max(self.Q[state].iteritems(), key=operator.itemgetter(1))[0]
        self.Q[state][best_action] += (1.0 - self.epsilon)
        #print Q[state]
        #print best_action
        return best_action
        #return random.randint(0, 1)


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.

        #state = self.construct_state(state)

        best_action = max(self.Q[state].iteritems(), key=operator.itemgetter(1))[0]
        #print self.Q[state]
        #print best_action
        return best_action


    def environment_variables(self):
        player_vel = range(-20, 20)

        cnt_py = 0
        cnt_ptop = 0
        cnt_pdist = 0
        player_y = []
        next_pipe_top_y = []
        next_pipe_dist_to_player = []
        for i in range(15):
            tmp = cnt_py + int(300 / 15)
            player_y.append((cnt_py, tmp))
            cnt_py = tmp

            tmp = cnt_ptop + int(250 / 15)
            next_pipe_top_y.append((cnt_ptop, tmp))
            cnt_ptop = tmp

            tmp = cnt_pdist + int(350 / 15)
            next_pipe_dist_to_player.append((cnt_pdist, tmp))
            cnt_pdist = tmp

        return player_vel, player_y, next_pipe_top_y, next_pipe_dist_to_player


    def environment_state_space(self, player_vel, player_y, next_pipe_top, next_pipe_dist):
        state_space = []
        for p_vel in player_vel:
            for p_y in player_y:
                for next_top in next_pipe_top:
                    for next_dist in next_pipe_dist:
                        state_space.append((p_vel, p_y, next_top, next_dist))

        return state_space


    def construct_state(self, state):
        p_y, next_top, next_dist = 0, 0, 0

        if state[1] < 0:
            p_y = self.environment['player_y'][0]
        else:
            for idx, item in enumerate(self.environment['player_y']):
                if item[0] <= state[1] < item[1]:
                    p_y = item
                    break

        if state[2] < 0:
            p_y = self.environment['next_pipe_top_y'][0]
        else:
            for idx, item in enumerate(self.environment['next_pipe_top_y']):
                if item[0] <= state[2] < item[1]:
                    next_top = item
                    break

        if state[3] < 0:
            next_dist = self.environment['next_pipe_dist_to_player'][0]
        else:
            for idx, item in enumerate(self.environment['next_pipe_dist_to_player']):
                if item[0] <= state[3] < item[1]:
                    next_dist = item
                    break

        return (state[0], p_y, next_top, next_dist)


    def _init_Q(self):
        Q = {}
        for state in self.environment_state_space(self.environment['player_vel'],
                                                  self.environment['player_y'],
                                                  self.environment['next_pipe_top_y'],
                                                  self.environment['next_pipe_dist_to_player']):
            Q[state] = {0: 0.0, 1: 0.0}
        return Q


    def save_Q(self):
        location = './Q/MC_Q.pkl'
        with open(location, 'wb') as file:
            pickle.dump(self.Q, file)



def plot_results(episodes, scores, save=False):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores)
    plt.show()



def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    discount_factor = 0.1


    #reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    n_episode_list = []
    score_list = []
    episode_list = []
    episode_count = 0
    score = 0
    while nb_episodes > episode_count:
        # pick an action
        # TODO: for training using agent.training_policy instead
        state = env.game.getGameState()
        # TODO: Cleanup
        state = agent.construct_state((state['player_vel'], state['player_y'], state['next_pipe_top_y'], state['next_pipe_dist_to_player']))
        action = agent.training_policy(state)#policy(state)
        #print action
        #if action == 1:
        #    print "TICK"

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        episode_list.append((state, action, reward))

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()

            # Find all (state, action) pairs we've visited in this episode
            # We convert each state to a tuple so that we can use it as a dict key
            sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode_list])
            for state, action in sa_in_episode:
                #if Q[state][1] > 0.0:
                #    print Q[state]
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next(i for i, x in enumerate(episode_list)
                                           if x[0] == state and x[1] == action)
                # Sum up all rewards since the first occurance
                G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode_list[first_occurence_idx:])])
                # Calculate average return for this state over all sampled episodes
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                agent.Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            score = 0
            env.init()
            game_on = True
            print('=========    Starting trained game   =========')
            while game_on:
                state = env.game.getGameState()
                state = (
                state['player_vel'], state['player_y'], state['next_pipe_top_y'], state['next_pipe_dist_to_player'])
                state = agent.construct_state(state)
                action = agent.policy(state, agent.Q)
                reward = env.act(env.getActionSet()[action])
                score += reward

                if env.game_over():
                    print("Score for the Game: ", score)
                    n_episode_list.append(episode_count)
                    score_list.append(score)
                    env.reset_game()
                    score = 0
                    game_on = False


            episode_count += 1
            score = 0

    print('=========    Starting trained game   =========')

    # Play 1 games after training
    games = 1
    score = 0
    env.init()
    while games > 0:
        state = env.game.getGameState()
        state = (state['player_vel'], state['player_y'], state['next_pipe_top_y'], state['next_pipe_dist_to_player'])
        state = agent.construct_state(state)
        action = agent.policy(state, agent.Q)
        reward = env.act(env.getActionSet()[action])
        score += reward

        if env.game_over():
            print("Score for the Game: ", score)
            env.reset_game()
            games -= 1
            score = 0

    agent.save_Q()
    plot_results(n_episode_list, score_list)

agent = FlappyAgent()
run_game(5, agent)
