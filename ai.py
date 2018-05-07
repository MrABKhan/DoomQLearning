import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


#Self Notes

# imp note no.1: Pytorch.2d images uses (chan x height x width), height x width must be equal or greater than the kernel size

 

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

import image_preprocessing
import experience_replay



class CNN(nn.Module):
    
    def __init__(self, number_actions):
#Constructor, used this because I can manipulate other doom envs with diff actoins

#inheritance of all the stuff from the nn module, basic OO concept

	#now to make the CNN, I will use 4 COnvulutional layers
	
# how it works

# inchannels is number of input chanells for a 2d image, that is grayscale in my case so it's 1 else 3 is for RGB image.

# outchannels is the number of output features that will get from the convo operation
# Documentation
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) 
#kernel_size aka window size


#how to decide the number of channels and everything = mera code,meri marzi principle :) just DIY
 
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)

        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
#self.convolayer4 = nn.Conv2d(in_channels=64, 			out_channels = 128, kernel_size =3)
	

#no to just flatten it and send it to an ann ( that is hidden layer first )
#class torch.nn.Linear(in_features, out_features, bias=True) from documentation


		# to the hidden layer I will send the number of neurons from the hidden layers via a flatten fashion

		# first hidden layer I will have arbitrary choice over the no. of nodes.

        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
#second layer will be the output layer and it's input will be from previous layer and output will be equal to the Q values aka the number of actions.

        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

#self.convolayer4 = nn.Conv2d(in_channels=64, 			out_channels = 128, kernel_size =3)
	

#no to just flatten it and send it to an ann ( that is hidden layer first )
#class torch.nn.Linear(in_features, out_features, bias=True) from documentation


		# to the hidden layer I will send the number of neurons from the hidden layers via a flatten fashion

		# first hidden layer I will have arbitrary choice over the no. of nodes.

#to calculate numberofneurons from the base doom 80x80 image

    def count_neurons(self, image_dim):
#random image 		with one chaneel and dimensions as the same as input image

		 # convert it into a pytorch var for 		procesing further

	#After this I will just apply the convolution on the base 		image, max pool it and maybe give it a stride and pass it 		through all of the convo's that I made in the model! also to 		use the activation function on them.
        x = Variable(torch.rand(1, *image_dim))

        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))

        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))

        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
#tmp = F.relu(F.max_pool2d(self.convolayer4(tmp), kernel_size = 3, stride = 2))
		#so now to flatten based on the Pytorch tut available 		in the Documentation, it is a 'trick'. Since X is an structure 		aka variable, I first check the data, then i veiw the data aka 		output it on an imaginary screen, and then use the size 		function to squish it into one dimension...so this is 		how it works. it's there in the documentation. Why pytorch why ?
        return x.data.view(1, -1).size(1)

# So now our actual input aka the signal is ready!!! we 	shaped the signal based on our CNN "MODEL"



#now to actually feed the image through the model~

    def forward(self, x):
#copy-pasting from above because it's all the same convulutions	

        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
#tmp = F.relu(F.max_pool2d(self.convolayer4(tmp), kernel_size = 3, stride = 2))
		

	#time to use that pytorch flatten trick!
        x = x.view(x.size(0), -1)
#after flattening it's time to send it to the fully connected laters

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#so the forward pass si done done dne
        return x


############################################################################################################

#So that's it for the model.

#Now to Finally take action based on the model!


class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
#Temp hypervar! it will scale the output of the model

    def forward(self, outputs):
#this will take output of the model and take some action based on that 

	#this iwill be a softmax

# we will find the probs first for each action! which are 7 in total! 
        probs = F.softmax(outputs * self.T) 
  
        actions = probs.multinomial()#actual softmax probability

        return actions


# now to make a singular class for all of the stuff above in one gaint function


class AI:

    def __init__(self, brain, body):
        self.brain = brain #initialize model

        self.body = body #initialize actor or the actuator

    def __call__(self, inputs):
#take input as image to np.array, then np.array to torch tensor, and then from torch tensor to torch variable so it can go through all of the convo2d
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
#got input
#send input to CNN
#output of Q values of CNN to actor aka actuator
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()




#################################################################


##For Turning image into grayscale and downsizing it!

doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
#for video recording! thank you gym!
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
#no. of actions so I can work on other doom envs also! :DDDD

no_of_actions = doom_env.action_space.n
 
# so no just plugit in the sigularity and lets go go go :D
mymodel = CNN(no_of_actions)
theactor = SoftmaxBody(T=1.0)
final = AI(brain = mymodel, body = theactor) 

###############


#calculating tht reward for the Nstep progression!
n_steps = experience_replay.NStepProgress(env = doom_env, ai = final, n_step = 10)

#sending the nstep obj to the replay memory inorder to effectively use it!

memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

#####################################################


##So now I would like to implement the Eligibility Trace Function as described in the tut.

# So download tut file for action replay ( piece of cake ), under stand and get n steps  and write the elibility trace ftn


#batch size because I'll be gettin them in batches.

def eligibility_trace(batch):
#so basically I will be getting batches of series of states, each series will be of n-steps for the algorightms
    gamma = 0.99
##gamma value becuz tut is taking it directly from papers aka tut
    inputs = []
#input lists aka the INPUT that is first state of the n-steps for all of the batches that I gave em.
    targets = []
#target lists will be the cumulative reward for all of the states except last state in a nth-stepth.  
    for series in batch:
# so according to pseudo algo in the paper and the tut, I will take the
# state in a series of steps and the last one.
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))

#pass it through the cnn to get it's q values.
        output = mymodel(input)
# so cumulative reward will be 0 if we reach the last state aka that state is done, or else it is the max of the Q-values for the non-terminal states, 
#output[1] becasue the structure contains the data values ( thanks pytorch :P ) and get it's max.


        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
# so according to the algo in the paper to go from the second last step to the first step, and add the cul reward in that series.


        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
# rest everythin is according to the algo, take the first state of the series and the predicted q values
#first state 
        state = series[0].state
#qvalue for the input state
        target = output[0].data
#the target associated to the first step of the series is equal to the cum reward!
        target[series[0].action] = cumul_reward
#now just to append the state to the batch
        inputs.append(state)
        targets.append(target)
#return the inputs and the targets using ***** pytorch.
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)



#moving avergae for the reward calculation, simple math func
class MA:
    def __init__(self, size):
#simple constructor
        self.list_of_rewards = []
        self.size = size



    def add(self, rewards):
        if isinstance(rewards, list):
#checks where reward is a list of rewards or a single reward because I will use it in the future.
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)

#moving avg of 100 steps sounds nice that i take 10 n-steps for 10 batches,
ma = MA(100)



## Ishouldkillmyself
#;dsf##################



##now to just configure the loss ftn and the optimizers
##I'll just take the one's readily available, dont ask questions too tired.
loss = nn.MSELoss()

optimizer = optim.Adam(mymodel.parameters(), lr=0.001 )


##just gonnna loop dat stuff for that epoch.

for ep in range(1,1000):
	#200 runs of 10 steps! using that run_steps ftn.
	memory.run_steps(200)
	#now to get random batches and learn on them hehhehehehe
	for batch in memory.sample_batch(128):


#run the eg on it to get the inputs and targets
		inputs, targets = eligibility_trace(batch)
#everyone say I hate Pytorch :D
		inputs, targets = Variable(inputs), Variable(targets)
#run it through the cnn again to get the predicted q values for error calc
		predictions = mymodel(inputs)

#calculate the loss ftn
		lossakaerror = loss(predictions, targets)
		#set the optimizer to zero to backpropagate the loss
		optimizer.zero_grad()
		#to do back prop and update weights through pytorch.
		lossakaerror.backward()
		optimizer.step()
# SOOOOO FINALLY! now to just print it out, but first the rewards.
	reward_steps = n_steps.rewards_steps() #new cum_reword for each Nstep

# add it to the moving avg.
	ma.add(reward_steps)
	avg_reward = ma.average()
	print("Epoch : %s, Average-Reward: %s" % (str(ep), str(avg_reward)))			
	if avg_reward >=1500:
		print("ML ka course zindabad, this took 63 hours to make and understand (also some very worth it 30$), so for the life of me please be kind. sigh Goodnight")
		break



