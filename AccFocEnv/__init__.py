"""Register AccFocEnv as a OpenAI Gym environment
"""

from gym.envs.registration import register

register(
    id='AccFocEnv-v0',
    entry_point='AccFocEnv.AccFocEnv:AccFocEnv',
)
