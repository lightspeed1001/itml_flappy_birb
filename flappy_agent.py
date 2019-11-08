from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
from statistics import mean
from collections import defaultdict
import time
import numpy

class FlappyAgentMC:
    def __init__(self, epsilon, discount, buckets):
        self.reward_for_state_action_pair = defaultdict(float) # Q - List of state/action pairs and their expected rewards
        self.returns_s_a = defaultdict(float) # Expected return for episode
        self.returns_sa_count = defaultdict(float)
        self.epsilon = epsilon
        self.episode = []
        self.discount = discount
        self.buckets = buckets
    
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
        self.episode.append((s1,a,r))
        if end:
            for s,a,r in self.episode[::-1]:
                pair = (s,a)
                first_occurance = next(i for i,x in enumerate(self.episode) if x[0] == s and x[1] == a)
                G = sum([x[2] * (self.discount ** i) for i,x in enumerate(self.episode[first_occurance:])])

                self.returns_s_a[pair] += G
                self.returns_sa_count[pair] += 1.0
                self.reward_for_state_action_pair[pair] = (self.returns_s_a[pair] / self.returns_sa_count[pair])
                
            self.episode = []



    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #To flap, or not to flap, that is the question
        action = 0
        #Exlore y/n?
        if random.random() >= self.epsilon:
            #call policy, no need to have the same code twice over.
            action = self.policy(state)
        else:
            action = random.randint(0,1)
        return action

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # If the reward for flapping is higher, we want to flap.
        if self.reward_for_state_action_pair[(state, 1)] > self.reward_for_state_action_pair[(state, 0)]:
            return 1
        else:
            return 0

    def discretize_state(self, state):
        distance_y = (state['next_pipe_top_y'] - state['player_y']) // self.buckets
        # player_y = state['player_y'] // self.buckets
        # pipe_y = state['next_pipe_top_y'] // self.buckets
        velocity = state['player_vel']
        pipe_dist = state['next_pipe_dist_to_player'] // self.buckets
        disc_state = (distance_y, velocity, pipe_dist)
        return disc_state

class FlappyAgentQL(FlappyAgentMC):
    def __init__(self, epsilon, learningRate, discount, buckets):
        super().__init__(epsilon, discount, buckets)
        self.learning_rate = learningRate

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
        pair = (s1,a)
        if pair not in self.reward_for_state_action_pair:
            self.reward_for_state_action_pair[pair] = r
        else:
            if not end:
                sPrime = self.policy(s2) #get action for s2
                self.reward_for_state_action_pair[pair] = self.reward_for_state_action_pair[pair] + self.learning_rate *(r + self.discount * self.reward_for_state_action_pair[(s2,sPrime)] - self.reward_for_state_action_pair[pair])
            else:
                self.reward_for_state_action_pair[pair] = self.reward_for_state_action_pair[pair] + self.learning_rate * (r - self.reward_for_state_action_pair[pair])



    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        if random.random() >= self.epsilon:
            action = self.policy(state)
        else:
            #Fair randomness for flapping
            if random.randint(1,10) == 10:
                action = 0
            else:
                action = 1
        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        if self.reward_for_state_action_pair[(state,0)] >= self.reward_for_state_action_pair[(state,1)]:
            action = 0
        else:
            action = 1
        return action

    def discretize_state(self, state):
        distance_y = (state['next_pipe_top_y'] - state['player_y']) // self.buckets
        # player_y = state['player_y'] // self.buckets
        # pipe_y = state['next_pipe_top_y'] // self.buckets
        velocity = state['player_vel']
        pipe_dist = state['next_pipe_dist_to_player'] // self.buckets
        #next_distance_y = (state['next_next_pipe_top_y'] - state['next_pipe_top_y']) // self.buckets
        disc_state = (distance_y, velocity, pipe_dist)

        return disc_state

class FlappyAgentLR(FlappyAgentMC):
    def __init__(self, epsilon, learningRate, discount, buckets,flapW, noopW):
        super().__init__(epsilon, discount, buckets)
        self.learning_rate = learningRate
        self.flapWeight = flapW
        self.noopWeight = noopW

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
        pair = (s1,a)
        sPrime = self.policy(s2) #get action for s2
        self.episode.append([s1,a,r])
        if end:
            G = 0
            for s,a,r in self.episode[::-1]:
                pair = (s,a)
                first_occurance = next(i for i,x in enumerate(self.episode) if x[0] == s and x[1] == a)
                G = sum([x[2] * (self.discount ** i) for i,x in enumerate(self.episode[first_occurance:])])
                if a == 0:
                    for i in range(4):
                        self.flapWeight[i] = self.flapWeight[i] + self.learning_rate * (G - self.flapWeight[i]) * self.reward_for_state_action_pair[pair]
                else:
                    for i in range(4):
                        self.noopWeight[i] = self.noopWeight[i] + self.learning_rate * (G - self.noopWeight[i]) * self.reward_for_state_action_pair[pair]
            self.episode = []

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        if random.random() >= self.epsilon:
            action = self.policy(state)
            pass
        else:
            action = random.randint(0,1)
        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        action = 1
        flap = numpy.dot(self.flapWeight,state)
        noop = numpy.dot(self.noopWeight,state)
        if flap > noop:
            action = 0
        return action

    def discretize_state(self, state):
        return (state['player_y'],state['player_vel'],state['next_pipe_top_y'],state['next_pipe_dist_to_player'])

class FlappyAgentQBest(FlappyAgentMC):
    def __init__(self, epsilon, learningRate, discount, buckets):
        super().__init__(epsilon, discount, buckets)
        self.learning_rate = learningRate

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
        pair = (s1,a)
        if pair not in self.reward_for_state_action_pair:
            self.reward_for_state_action_pair[pair] = r # start with 0 reward for each state if it does not exist
        else:
            if not end:
                sPrime = self.policy(s2) #get action for s2
                self.reward_for_state_action_pair[pair] = self.reward_for_state_action_pair[pair] + self.learning_rate *(r + self.discount * self.reward_for_state_action_pair[(s2,sPrime)] - self.reward_for_state_action_pair[pair])
            else:
                self.reward_for_state_action_pair[pair] = self.reward_for_state_action_pair[pair] + self.learning_rate * (r - self.reward_for_state_action_pair[pair])



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
        if random.random() >= self.epsilon:
            action = self.policy(state)
            pass
        else:
            #Fair randomness for flapping
            if random.randint(1,10) == 10:
                action = 0
            else:
                action = 1
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
        if self.reward_for_state_action_pair[(state,0)] >= self.reward_for_state_action_pair[(state,1)]:
            action = 0
        else:
            action = 1
        return action

    def discretize_state(self, state):
        distance_y = (state['next_pipe_top_y'] - state['player_y']) // self.buckets

        velocity = state['player_vel']

        pipe_dist = state['next_pipe_dist_to_player'] // self.buckets

        disc_state = (distance_y, velocity, pipe_dist)

        return disc_state

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()
    
    eps_run = 0
    highscore = 0
    score = 0
    next_state = None
    all_the_scores = []
    previous_100_scores = []
    bonusCounter = 0
    training = True
    start_time = time.time()
    frames = 0

    while nb_episodes > 0:
        if next_state is None:
            # Initial state
            state = agent.discretize_state(env.game.getGameState())
        else:
            # The s2 from last iteration
            state = next_state
        # Get an action and reward
        if training:
            action = agent.training_policy(state)
        else:
            action = agent.policy(state)
        reward = env.act(env.getActionSet()[action])

        # Get s2
        next_state = agent.discretize_state(env.game.getGameState())

        # call observe state
        agent.observe(state,action,reward,next_state,env.game_over())

        if reward > 0:
            # Just want to see the number of pipes we got through
            score += reward

        timediff = time.time() - start_time
        frames += 1
        # reset the environment if the game is over
        if (timediff >= TIME_LIMIT or frames >= FRAME_LIMIT) and training:
            print("Time or frame limit reached! Moving on to testing.")
            training = not training
            nb_episodes = 50
            all_the_scores = []
            previous_100_scores = []
            eps_run = 0
            env.reset_game()
            continue
        
        if env.game_over() or score > 149: # stop at 150 pipes, good birbs are time consuming
            if nb_episodes % 50 == 1:
                print("episodes remaing {}".format(nb_episodes))
                # env.display_screen = True
                # env.force_fps = False
                if training:
                    #Starting with initial high randomness and learning, dropping it over time gives us a better learning curve
                    if agent.epsilon > 0.1:
                        agent.epsilon *= 0.8
                        #if agent.epsilon <= 0.1:
                            #print("Epsilon limit reached")
                    if agent.learning_rate > 0.25:
                        agent.learning_rate *= 0.95
                        #if agent.learning_rate <= 0.25:
                            #print("Learning Rate limit reached")

            if score > highscore:
                highscore = score
            all_the_scores.append(score)
            previous_100_scores.append(score)
            if score > 0:
                print("{}; score for this episode: {}; highscore: {}; avg: {}; avg100: {}".format("training" if training else "testing",score, highscore, mean(all_the_scores), mean(previous_100_scores)))
                if not training and score < 50:
                    bonusCounter += 1
                    print("Failed {} tests for bonus thus far".format(bonusCounter))
            if len(previous_100_scores) >= 100:
                previous_100_scores.remove(previous_100_scores[0])
            if not training:
                with open(FILENAME, "a") as f:
                    f.write("{},{}\n".format(eps_run, score))#,"train" if training else "test", mean(all_the_scores), mean(previous_100_scores)))
            env.reset_game()
            next_state = None
            nb_episodes -= 1
            eps_run += 1
            score = 0

FILENAME = "QbestFinal2.csv"
TIME_LIMIT = 60 * 60 # one hour
FRAME_LIMIT = 1000000 # 1m frames
with open(FILENAME, "w+") as f:
    f.write("episode,score\n") #,type,average,average100\n")
# agent = FlappyAgentMC(epsilon=0.1, discount=0.99, buckets=15)
# agent = FlappyAgentQL(epsilon=1.0,learningRate=0.25, discount=0.9, buckets=30)
# agent = FlappyAgentLR(epsilon=1.0, learningRate=0.25, discount=1.0, buckets=15, flapW=[0,0,0,0], noopW=[0,0,0,0])
agent = FlappyAgentQBest(epsilon=1.0, learningRate=0.9, discount=0.99, buckets=30)
run_game(100000, agent)
