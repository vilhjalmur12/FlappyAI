from unittest import TestCase
import os
import numpy as np
from collections import defaultdict
import operator
import matplotlib.pyplot as plt

#from flappy_agent_mc import FlappyAgent

os.chdir('../../')

class test_flappy_mc(TestCase):

    def test_env_list(self):
        player_vel = range(-10, 5)

        tmp = 0
        player_y = [tmp]
        for cnt in range(14):
            tmp = tmp + int(250/13)
            player_y.append(tmp)

        tmp = 0
        next_pipe_top_y = [tmp]
        for cnt in range(14):
            tmp = tmp + int(170/14)
            next_pipe_top_y.append(tmp)

        tmp = 0
        next_pipe_dist_to_player = [tmp]
        for cnt in range(14):
            tmp = tmp + int(350 / 14)
            next_pipe_dist_to_player.append(tmp)

        name = ''

    def test_state_space(self):
        player_y = [ '0', '1', '2' ]
        next_pipe = [ 'one', 'two', 'three' ]
        velocity = [ 'slow', 'medium', 'fast' ]

        state_list = []

        for p in player_y:
            for pipe in next_pipe:
                for vel in velocity:
                    state_list.append((p, pipe, vel))

        name = ''

    def test_action_space(self):
        t = defaultdict(lambda: np.zeros(2))



        name = ''

    def environment_state_space_OLD(self):
        player_vel = range(-10, 5)

        tmp = 0
        player_y = [tmp]
        for cnt in range(14):
            tmp = tmp + int(250 / 13)
            player_y.append(tmp)

        tmp = 0
        next_pipe_top_y = [tmp]
        for cnt in range(14):
            tmp = tmp + int(170 / 14)
            next_pipe_top_y.append(tmp)

        tmp = 0
        next_pipe_dist_to_player = [tmp]
        for cnt in range(14):
            tmp = tmp + int(350 / 14)
            next_pipe_dist_to_player.append(tmp)

        state_space = []

        for vel in player_vel:
            for p_y in player_y:
                for next_top in next_pipe_top_y:
                    for next_dist in next_pipe_dist_to_player:
                        state_space.append((vel, p_y, next_top, next_dist))

        return state_space


    def test_environment_state_space(self):
        player_vel = range(-10, 5)

        cnt_py = 0
        cnt_ptop = 0
        cnt_pdist = 0
        player_y = []
        next_pipe_top_y = []
        next_pipe_dist_to_player = []
        for i in range(15):
            tmp = cnt_py + int(280/15)
            player_y.append((cnt_py, tmp))
            cnt_py = tmp

            tmp = cnt_ptop + int(200 / 15)
            next_pipe_top_y.append((cnt_ptop, tmp))
            cnt_ptop = tmp

            tmp = cnt_pdist + int(350 / 15)
            next_pipe_dist_to_player.append((cnt_pdist, tmp))
            cnt_pdist = tmp

        state_space = []
        for p_vel in player_vel:
            for p_y in player_y:
                for next_top in next_pipe_top_y:
                    for next_dist in next_pipe_dist_to_player:
                        state_space.append((p_vel, p_y, next_top, next_dist))

        return state_space


    def test_create_state_action_dict(self):

        Q = {}
        for state in self.environment_state_space():
            Q[state] = { 0: 0.0, 1: 0.0 }



    def test_argmax_Q(self):
        Q = {
            (-2, 3, 4): { 0: 0.0, 1: 0.1},
            (1, 23, 16): { 0: 0.2, 1: 0.0 }
        }

        test_state = (1, 23, 16)
        tes = max(Q[test_state].iteritems(), key=operator.itemgetter(1))[0]

        name = ''

    def test_make_policy(self):
        A = np.zeros(2, dtype=float) * 0.1 / 2

        name = ''


    def test_should_choose_tick(self):
        state = (2, -1, 5)
        Q = { state: { 0: -5.0, 1: 0.0 } }

        best_action = max(Q[state].iteritems(), key=operator.itemgetter(1))[0]

        self.assertEqual(best_action, 1)


    def test_construct_state(self):

        state = (2, 230, 100, 120)

        player_vel, player_y, next_pipe_top_y, next_pipe_dist_to_player = self.environment_state_space()
        p_y, next_top, next_dist = 0,0,0

        for idx, item in enumerate(player_y):
            if item[0] < state[1] < item[1]:
                p_y = item
                break

        for idx, item in enumerate(next_pipe_top_y):
            if item[0] < state[2] < item[1]:
                next_top = item
                break

        for idx, item in enumerate(next_pipe_dist_to_player):
            if item[0] < state[3] < item[1]:
                next_dist = item
                break




        ret_val = (state[0], p_y, next_top, next_dist)
        name = ''


    def test_plotting(self):
        episodes = [1, 2, 3, 4, 5]
        scores = [0, 0, 3, 4, 6]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, scores, 'r')
        plt.grid(True)
        #plt.xticks()
        plt.title('Scores over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.show()






