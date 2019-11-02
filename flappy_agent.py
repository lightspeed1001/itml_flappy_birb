from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
from statistics import mean
from collections import defaultdict


class FlappyAgentMC:
    def __init__(self, epsilon, learningRate, discount, buckets):
        # TODO: you may need to do some initialization for your agent here
        self.reward_for_state_action_pair = {} # Q - List of state/action pairs and their expected rewards
        self.returns_s_a = defaultdict(float) # Expected return for episode
        self.returns_sa_count = defaultdict(float)
        self.epsilon = epsilon 
        self.learning_rate = learningRate
        self.episode = []
        self.discount = discount
        self.buckets = buckets

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
            for s,a,r in self.episode:
                pair = (s,a)
                first_occurance = next(i for i,x in enumerate(self.episode) if x[0] == s and x[1] == a)
                G = sum([x[2] * (self.discount ** i) for i,x in enumerate(self.episode[first_occurance:])])

                self.returns_s_a[pair] += G
                self.returns_sa_count[pair] += 1.0
                self.reward_for_state_action_pair[pair] = (self.returns_s_a[pair] / self.returns_sa_count[pair])
            self.episode = []
            """ for s,a,r in self.episode[::-1]: #run through the episodes from last to first
                pair = (s,a)
                reward_for_state = r + self.discount * reward_for_state
                if (s,a) in self.reward_for_state_action_pair.keys():
                    self.reward_for_state_action_pair[(s,a)] = self.reward_for_state_action_pair[(s,a)] + self.learning_rate*(reward_for_state-self.reward_for_state_action_pair[(s,a)])
                else:
                    self.reward_for_state_action_pair[(s,a)] = reward_for_state
                    
                self.returns_s_a[pair] += reward_for_state
                
                self.reward_for_state_action_pair[pair] = (sum([x for x in self.returns_s_a.values()]) / len(self.returns_s_a))
                
            self.episode = [] """

    

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
            if (state, 1) in self.reward_for_state_action_pair.keys():
                no_flap = self.reward_for_state_action_pair[(state,1)]
            else:
                no_flap = 0 
            if (state, 0) in self.reward_for_state_action_pair.keys():
                flap = self.reward_for_state_action_pair[(state, 0)]
            else:
                flap = 0
            
            if flap > no_flap:
                action = 0 # don't flap
            else:
                action = 1 # flap
        else:
            action = random.randint(0,1)

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
    total_score = 0
    avg_score = 0
    
    while nb_episodes > 0:
        # Generate an episode using policy
        # pick an action
        state = agent.discretize_state(env.game.getGameState())
        #print(state)
        # TODO: for training using agent.training_policy instead
        action = agent.training_policy(state)
        # TODO Discretize state
        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)
        # call observe state
        agent.observe(state,action,reward,None,env.game_over())
        # TODO: for training let the agent observe the current state transition
        if reward > 0:
            score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            if(nb_episodes % 100 == 1):
                env.display_screen = True
                env.force_fps = False
            else:
                env.display_screen = False
                env.force_fps = True
            
            if score > highscore:
                highscore = score
            if score > 0:
                total_score += score
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


    # TODO Test the found policy here
    print("Highscore: %d" % highscore)

agent = FlappyAgentMC(epsilon=0.0001,learningRate=0.1, discount=0.99, buckets=30)
run_game(1000, agent)
