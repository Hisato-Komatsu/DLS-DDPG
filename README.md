# DLS-DDPG
Source codes for "Application of linear regression method to the deep reinforcement learning in continuous action cases".

The versions of Python and the key libraries used in the calculations in this paper are as follows: <br>
Python               3.11.10 <br>
gymnasium            1.0.0 <br>
mujoco               3.2.4 <br>
numpy                1.26.0 <br>
scipy                1.14.1 <br>
torch                2.5.0+cpu <br>

# credits
[Added on 2025-08-10 to increase transparency.]
This code was influenced by multiple open-source implementations, including:

- https://github.com/seolhokim/Mujoco-Pytorch (MIT license)

In particular, the function `memory_sample` is based on the following implementation:

https://qiita.com/Rowing0914/items/eeba790401bcaf2c723c

and the soft target update is implemented based on the following code:

https://github.com/ghliu/pytorch-ddpg (Apache-2.0 license)

We sincerely appreciate the contributions of the original authors and acknowledge their impact on this work.
