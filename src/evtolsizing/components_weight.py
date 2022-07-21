import numpy as np
import openmdao.api as om

# ------------------------------
# weight estimation components
# ------------------------------

class PropulsionWeight(om.ExplicitComponent):
    """
    computes weights of rotor, motor and ESC.
    Inputs: rotor radius, max power (here assumed to be the same as hovering power)
    Outputs: rotor weight, motor&ESC weight
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of rotosr')

    def setup(self):
        n_UAVs = self.options['n_UAVs']

        self.add_input('UAVs|rotor_radius', val=0.3 * np.ones(n_UAVs), units='inch', desc='rotor radius')
        self.add_input('power_max', shape=(n_UAVs,), units='hp', desc='max power')
        self.add_output('UAVs|W_rotor_all', shape=(n_UAVs,), units='kg', desc='rotor weight, sum of all rotors')
        self.add_output('UAVs|W_motor_all', shape=(n_UAVs,), units='lb', desc='motor & ESC weight, sum of all')
        self.declare_partials('UAVs|W_rotor_all', 'UAVs|rotor_radius', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs))
        self.declare_partials('UAVs|W_motor_all', 'power_max', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs))

        self.power_overhead_factor = 1.5   # installed power = overhead factor * max power output

    def compute(self, inputs, outputs):
        n_rotor = self.options['n_rotor']
        radius_in = inputs['UAVs|rotor_radius']
        power_max_each = self.power_overhead_factor * inputs['power_max'] / n_rotor     # max power for each rotor.
        W_rotor = 0.1207 * (2 * radius_in)**2 - 0.5122 * (2 * radius_in)   # [g], propeller. Make sure radius in [inch]
        W_motor = (0.412 + 0.591) * power_max_each    # [kg], motor + ESC. Make sure power in [hp]
        outputs['UAVs|W_rotor_all'] = W_rotor / 1000 * n_rotor   # [kg]
        outputs['UAVs|W_motor_all'] = W_motor * n_rotor        # [lb]

    def compute_partials(self, inputs, partials):
        n_rotor = self.options['n_rotor']
        radius_in = inputs['UAVs|rotor_radius']
        d_power_max_each_dp = self.power_overhead_factor / n_rotor
        d_Wrotor_d_r = 0.1207 * 4 * 2 * radius_in - 0.5122 * 2
        partials['UAVs|W_rotor_all', 'UAVs|rotor_radius'] = n_rotor / 1000 * d_Wrotor_d_r
        partials['UAVs|W_motor_all', 'power_max'] = n_rotor * (0.412 + 0.591) * d_power_max_each_dp


class WingWeight(om.ExplicitComponent):
    """
    computes wing weight given the wing area
    Inputs: wing area
    Outputs: wing weight
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')

    def setup(self):
        n_UAVs = self.options['n_UAVs']
        self.add_input('UAVs|S_wing', shape=(n_UAVs,), units='m**2', desc='wing area')
        self.add_output('UAVs|W_wing', shape=(n_UAVs,), units='kg', desc='wing weight')
        self.declare_partials('*', '*', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs))

    def compute(self, inputs, outputs):
        outputs['UAVs|W_wing'] = -0.08017 + 2.2854 * inputs['UAVs|S_wing']

    def compute_partials(self, inputs, partials):
        partials['UAVs|W_wing', 'UAVs|S_wing'] = 2.2854
