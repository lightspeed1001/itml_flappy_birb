from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
from statistics import mean
from collections import defaultdict

class FlappyAgentMC:
    def __init__(self, epsilon, learningRate, discount, buckets):
        # TODO: you may need to do some initialization for your agent here
        self.reward_for_state_action_pair = defaultdict(float) # Q - List of state/action pairs and their expected rewards
        self.returns_s_a = defaultdict(float) # Expected return for episode
        self.returns_sa_count = defaultdict(float)
        self.epsilon = epsilon 
        self.learning_rate = learningRate
        self.episode = []
        self.discount = discount
        self.buckets = buckets
        self.previous_action = 0
        
        # halda utanum states í þessu episode
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0,}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        
        # For each pair s, a appearing in the episode:
        # G <- return following the first occurrence of s, a -- ???
        # Append G to Returns(s, a) -- ???
        # Q(s, a) <- average(Returns(s, a)) -- Q(s,a) er reward fyrir state/action pair

        # For each s in the episode:
        # A* = max(Q(s,a)) -- Finna max reward fyrir state/action pair
        # for all a in A(s) 
        # update policy? But how?

        # Hvað er A(s)? Action for a state? 
        self.episode.append((s1,a,r))
        if end:
            reward_for_state = 0
            for s,a,r in self.episode[::-1]:
                pair = (s,a)
                first_occurance = next(i for i,x in enumerate(self.episode) if x[0] == s and x[1] == a)
                G = sum([x[2] * (self.discount ** i) for i,x in enumerate(self.episode[first_occurance:])])

                self.returns_s_a[pair] += G
                self.returns_sa_count[pair] += 1.0
                self.reward_for_state_action_pair[pair] = (self.returns_s_a[pair] / self.returns_sa_count[pair])
                
                # idk lol
                # source: https://github.com/chncyhn/flappybird-qlearning-bot/blob/master/src/bot.py
                # max_reward = None
                # for s1, a2 in self.reward_for_state_action_pair.keys():
                #     if max_reward == None:
                #         max_reward = self.reward_for_state_action_pair[(s1, a2)]
                #     elif s1 == s:
                #         reward = self.reward_for_state_action_pair[(s1, a2)]
                #         if reward > max_reward:
                #             max_reward = reward
                # self.reward_for_state_action_pair[pair] = (1-self.epsilon) * (self.reward_for_state_action_pair[pair]) + \
                #                        self.epsilon * ( r + self.discount*max_reward)

                    
            self.episode = []

    

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        
        # Generate an episode using policy

        """
        if state in self.reward_for_state_action_pair:
            do a good thing?
        else:
            do a random thing?
        """
        #To flap, or not to flap, that is the question
        action = 0
        #Exlore y/n?
        if random.random() >= self.epsilon:
            if self.reward_for_state_action_pair[(state, 1)] > self.reward_for_state_action_pair[(state, 0)]:
                action = 1
            else:
                action = 0
            
            # if flap_reward > noflap_reward:
            #     action = 1
            # else:
            #     action = 0

            # if (state, 1) in self.reward_for_state_action_pair.keys():
            #     no_flap = self.reward_for_state_action_pair[(state,1)]
            # else:
            #     no_flap = 0 
            # if (state, 0) in self.reward_for_state_action_pair.keys():
            #     flap = self.reward_for_state_action_pair[(state, 0)]
            # else:
            #     flap = 0
            
            # if flap > no_flap:
            #     action = 0 # don't flap
            # else:
            #     action = 1 # flap
        else:
            action = random.randint(0,1)
        self.previous_action = action
        return action

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 
    
    def discretize_state(self, state):
        distance_y = (state['next_pipe_top_y'] - state['player_y']) // self.buckets
        velocity = state['player_vel']
        pipe_dist = state['next_pipe_dist_to_player'] // self.buckets
        disc_state = (distance_y, velocity, pipe_dist)#, self.previous_action)
        return disc_state

class FlappyAgentQL(FlappyAgentMC):
    def __init__(self, epsilon, learningRate, discount, buckets):
        super().__init__(epsilon, learningRate, discount, buckets)

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0,}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        if end:
            pass
            #learn from it

    

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        
        # Generate an episode using policy

        """
        if state in self.reward_for_state_action_pair:
            do a good thing?
        else:
            do a random thing?
        """


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 
    
    def discretize_state(self, state):
        #TODO: change this for QL
        distance_y = state['next_pipe_top_y'] - state['player_y']
        disc_state = (distance_y // self.buckets,
                state['player_vel'],
                state['next_pipe_dist_to_player'] // self.buckets)
        return disc_state

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = agent.reward_values() #{"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()
    highscore = 0
    score = 0
    next_state = None
    all_the_scores = []
    previous_100_scores = []
    while nb_episodes > 0:
        # Generate an episode using policy
        # pick an action
        if next_state is None:
            state = agent.discretize_state(env.game.getGameState())
        else:
            state = next_state
        #print(state)
        # TODO: for training using agent.training_policy instead
        action = agent.training_policy(state)
        # TODO Discretize state
        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)
        # call observe state
        next_state = agent.discretize_state(env.game.getGameState())
        agent.observe(state,action,reward,next_state,env.game_over())
        # TODO: for training let the agent observe the current state transition
        if reward > 0:
            score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            if(nb_episodes % 250 == 1):
                print("episodes remaing {}".format(nb_episodes))
                env.display_screen = True
                env.force_fps = False
                # agent.epsilon /= 2
            else:
                env.display_screen = False
                env.force_fps = True
            
            if score > highscore:
                highscore = score
            if score > 0:
                # print("score for this episode: {}; highscore: {}; episodes remaining: {}".format(score, highscore, nb_episodes))
                print("score for this episode: {}; highscore: {}; avg: {}; avg100: {}".format(score, highscore, mean(all_the_scores), mean(previous_100_scores)))
            all_the_scores.append(score)
            previous_100_scores.append(score)
            if len(previous_100_scores) >= 100:
                previous_100_scores.remove(previous_100_scores[0])
            env.reset_game()
            next_state = None
            nb_episodes -= 1
            score = 0


    # TODO Test the found policy here
    print("Highscore: %d" % highscore)
# epsilon 0.001, discount 0.99, buckets 30 = highscore: 26.0; avg: 1.636963696369637; avg100: 3.515151515151515
agent = FlappyAgentMC(epsilon=0.001,learningRate=0.1, discount=0.99, buckets=30)
# agent = FlappyAgentQL(epsilon=0.001,learningRate=0.1, discount=0.99, buckets=30)
run_game(10000, agent)
