"""This module implements different longitudinal vehicle models and needed subblasses."""
import numpy as np


def my_vehicle_model(*args, **kwargs):
    """Creates longitudinal vehicle model.

    Args:
        otps: Namespace object with options.

    Returns:
        Vehicle model object

    Raises:
        ValueError: If specified vehicle model is unknown.
    """
    vm = kwargs["opts"].vehicle_model
    if vm == "Simple":
        return SimpleVehicle(*args, T=0.5, **kwargs)
    else:
        raise ValueError(f"Vehicle model '{vm}' is unknown!")


class Integrator:
    """Implements a discrete linear time-invariant integrator for fixed time steps.
    """

    def __init__(self, dt=0.01, K=1, init=0.0, min_value=-np.inf, max_value=np.inf):
        """Initialization of parameters

        Args:
            dt: Interval in seconds.
            init: Initial value after reset.
            K: Gain.
            min_value: Enables clamping of the output to a minimum value.
            max_value: Enables calmping of the output to a maximum value.
        """
        self.dt = dt
        self.init_value = init
        self.value = init
        self.K = K
        self.min_value = min_value
        self.max_value = max_value

    def reset(self):
        """Resets value to initial value
        """
        self.value = self.init_value

    def __call__(self, value):
        """Integrates over input values.

        Args:
            value: Input value.

        Returns:
            Result of integration.
        """
        self.value += self.dt * value * self.K
        self.value = max(self.min_value, min(self.max_value, self.value))
        return self.value

    def get(self):
        """Returns current value of integrator.

        Returns:
            Current value of integrator.
        """
        return self.value


class PT1:
    """Implements a discrete PT1-filter for fixed time steps"""

    def __init__(self, dt=0.01, T=None, K=1, init=0.0):
        """Initialization of parameters

        Args:
            dt: Interval in seconds.
            T: Time constant in seconds. Needs to be specified.
            K: Gain.
            init: Initial value after reset.

        Raises:
            ValueError: If no time constant is specified.
        """
        self.dt = dt
        self.init_value = init
        self.value = init
        self.K = K
        self.T = T
        if T is None:
            raise ValueError("You have to specify a time constant!")
        self.T_star = 1.0 / (self.T / self.dt + 1)

    def reset(self):
        """Resets value to initial value"""
        self.value = self.init_value

    def __call__(self, value):
        """Filters input signal.

        Args:
            value: Input value.

        Returns:
            Result of filtering.
        """
        self.value = self.T_star * (self.K * value - self.value) + self.value
        return self.value

    def get(self):
        """Returns current value of filter.

        Returns:
            Current value of filter.
        """
        return self.value


class GradientLimiter:
    """Implements a gradient limiter for fixed time steps."""

    def __init__(self, dt=0.01, gradient_limit=5):
        """Initialization of parameters

        Args:
            dt: Interval in seconds.
            gradient_limit: Gradient limit.
        """
        self.change_per_step = gradient_limit * dt
        self.value = None

    def reset(self):
        """Resets internal value"""
        self.value = None

    def __call__(self, value):
        """Limits the gradient of the input signal.

        Args:
            value: Input value.

        Returns:
            Gradient limited signal.
        """
        # Assume first call to be steady state condition
        if self.value is None:
            self.value = value
            return value

        if value > self.value + self.change_per_step:
            self.value += self.change_per_step
        elif value < self.value - self.change_per_step:
            self.value -= self.change_per_step
        else:
            self.value = value

        return self.value

    def get(self):
        """Returns gradient limited signal

        Returns:
            Gradient limited signal
        """
        return self.value


class SimpleVehicle:
    """Simple longitudinal vehicle model.

    The input is the desired acceleration, which is filtered with a low-pass
    filter and integrated twice to obtain the actual acceleration, velocity and
    position of the vehicle.
    """

    def __init__(self, dt=0.1, T=0.5, K=1., a_init=0, v_init=0, x_init=0, allow_v_negative=True, opts={}):
        """Initialization of parameters

        Args:
            dt: Interval in seconds.
            T: Time constant of PT1-filter.
            K: Gain of PT1-filter.
            a_init: Initial longitudinal acceleration of the vehicle.
            v_init: Initial longitudinal velocity of the vehicle.
            x_init: Initial longitudinal position of the vehicle.
            allow_v_negative: If set to False, the vehicle will not drive backwards.
        """
        self.PT1 = PT1(dt=dt, T=T, init=a_init, K=K)
        if allow_v_negative:
            self.I1 = Integrator(dt=dt, init=v_init)
        else:
            self.I1 = Integrator(dt=dt, init=v_init, min_value=0)
        self.I2 = Integrator(dt=dt, init=x_init)
        self.allow_v_negative = allow_v_negative

    def step(self, a):
        """Simulates one time step.

        Args:
            a: Desired acceleration.

        Returns:
            a: Actual longitudinal acceleration.
            v: Actual longitudinal velocity.
            x: Actual longitudinal position.
        """
        a = self.PT1(a)
        v = self.I1(a)
        x = self.I2(v)

        if not self.allow_v_negative and a < 0 and v <= 0:
            a = 0

        return a, v, x

    def reset(self):
        """Resets all values to initial values.
        """
        self.I1.reset()
        self.PT1.reset()
        self.I2.reset()

    def set_v_init(self, v_init):
        """Sets the initial velocity.

        Args:
            v_init: Initial longitudinal velocity.
        """
        self.v_init = v_init
        self.I1.init_value = v_init

    def set_x_init(self, x_init):
        """Sets the initial position.

        Args:
            v_init: Initial longitudinal position.
        """
        self.x_init = x_init
        self.I2.init_value = x_init

    def set_v(self, v):
        """Sets the current velocity.

        Args:
            v: Current longitudinal velocity.
        """
        self.I1.value = v

    def get_v(self):
        """Gets the current velocity.

        Returns:
            Current longitudinal velocity.
        """
        return self.I1.value

    def set_s(self, s):
        """Sets the current position.

        Args:
            v: Current longitudinal position.
        """
        self.I2.value = s

    def get_s(self):
        """Gets the current position.

        Returns:
            Current longitudinal position.
        """
        return self.I2.value
