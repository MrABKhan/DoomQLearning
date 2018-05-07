# Playing Doom using Reinforcement Learning

## Introduction
Doom is a FPS game released in 1993. During the recent developments in AI and Machine Learning, it has become a trail of sorts for new machine learning algorithms, reason being that is varied, complex and with alot of player freedom and possibilities. This translates to a huge state-space and different action-spaces based on the levels.
  Currently http://vizdoom.cs.put.edu.pl/competition-cig-2017/results VIZDOOM cig is the "AI world championship" for which AI plays Doom best!

## Tools
1. OpenAIGym (0.9.5)
2. Pytorch 0.1.12


### Justifications for using the above  
  I used an older version of OpenAIGym as the newer version of Gym didnot include the Doom envs are they were depreceted and now being managed by https://github.com/ppaquette/gym-doom. This itself uses an older version of VizDoom. I could've used the newer Vizdoom API but due to time constraints of the semester, I could not.
  Since this is a Visual Env, as in I get my input in frames rather than variables of ram indicating score,reward,life etc, thus I am using a Convolutional Deep NN to find the Q-Values from the input. 
  Pytorch used to work great on my older laptop with a gpu so I used that, it was quicker due to CUDA on the lappy plus it I could install it easily with no errors or impedances.
  
## Scope
In my current implementation of Doom, I have limited it to using the "Doom-Corridor-v0" on the older version of OpenAIGym, due to the fact that other env's available on DOOM require a hell of a lot of training, thus this is a the simplest one I could do in reasonable time.


## Reinforcement Learning Techniques implemented
1. Deep Q Learning using convolutional neural networks ( because cnn's are really good in extracting features from images )
2. Memory Replay aka SARS ( State, Action, Reward, Next_State )
3. N-Step Q-Learning

All of the above concepts are explained in a simplified fashion in the Slides that I have attached in this repo.
