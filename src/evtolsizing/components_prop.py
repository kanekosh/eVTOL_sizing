import numpy as np
import openmdao.api as om

from evtolsizing.components_aero import WingedCruiseDrag, BodyDrag
from evtolsizing.utils import SoftMax

# ------------------------------
# Power estimation components
# ------------------------------

class PowerHover(om.ExplicitComponent):
    """
    computes the power required in hover
    Inputs: UAV weight, rotor radius
    Outputs: power required in hover, thrust of each rotor in hover
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of rotors')
        self.options.declare('hover_FM', types=float, desc='hover figure of merit')
        self.options.declare('rho_air', default=1.225, desc='air density')

    def setup(self):
        n_UAVs = self.options['n_UAVs']

        self.add_input('UAVs|W_total', shape=(n_UAVs,), units='kg', desc='total weight (MTOW)')
        self.add_input('UAVs|rotor_radius', val=0.3 * np.ones(n_UAVs), units='m', desc='rotor radius')
        self.add_output('power_hover', shape=(n_UAVs,), units='W', desc='power required in hover, after considering the prop efficiency')
        self.add_output('thrust_each', shape=(n_UAVs,), units='N', desc='thrust of each rotor in hover')
        self.declare_partials('*', '*', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs))

        self.gravity = 9.81

    def compute(self, inputs, outputs):
        n_rotor = self.options['n_rotor']
        efficiency = self.options['hover_FM']
        rho_air = self.options['rho_air']
        W_total = inputs['UAVs|W_total']
        rotor_radius = inputs['UAVs|rotor_radius']

        disc_area = np.pi * rotor_radius**2
        # power consumption (power needed to output from battery consumption, considering the battery and prop efficiency.)
        outputs['power_hover'] = (W_total**1.5 * np.sqrt(self.gravity**3 / (2 * rho_air * n_rotor * disc_area))) / efficiency
        outputs['thrust_each'] = W_total * self.gravity / self.options['n_rotor']
    
    def compute_partials(self, inputs, partials):
        n_rotor = self.options['n_rotor']
        efficiency = self.options['hover_FM']
        rho_air = self.options['rho_air']
        W_total = inputs['UAVs|W_total']
        rotor_radius = inputs['UAVs|rotor_radius']

        disc_area = np.pi * rotor_radius**2
        d_disc_area_d_radius = 2 * np.pi * rotor_radius

        partials['power_hover', 'UAVs|W_total'] = 1.5 * W_total**0.5 * np.sqrt(self.gravity**3 / (2 * rho_air * n_rotor * disc_area)) / efficiency
        partials['power_hover', 'UAVs|rotor_radius'] = W_total**1.5 * np.sqrt(self.gravity**3 / (2 * rho_air * n_rotor)) / efficiency * (-0.5 * disc_area**(-1.5)) * d_disc_area_d_radius
        partials['thrust_each', 'UAVs|W_total'] = self.gravity / self.options['n_rotor']


class PowerForwardEdgewise(om.Group):
    """
    computes the power required in edgewise forward flight (cruise of wingless multirotor)
    Inputs: UAV weight, cruise speed, rotor radius, edgewise advance ratio mu, rotor tilt angle
    Outputs: power (sum of all rotors), thrust of each rotor
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of lifting rotors')
        self.options.declare('hover_FM', types=float, desc='hover figure of merit')
        self.options.declare('rotor_sigma', types=float, desc='rotor solidity')
        self.options.declare('rho_air', default=1.225, desc='air density')

    def setup(self):
        n_UAVs = self.options['n_UAVs']
        n_rotor = self.options['n_rotor']
        rho_air = self.options['rho_air']

        # drag in cruise
        self.add_subsystem('body_drag', BodyDrag(n_UAVs=n_UAVs), promotes_inputs=['*'], promotes_outputs=['drag'])    # body drag is the only source of the drag
        
        # thrust required for trim in cruise
        self.add_subsystem('trim', MultiRotorTrim(n_UAVs=n_UAVs), promotes_inputs=['*'], promotes_outputs=[('thrust', 'thrust_all'), 'body|sin_beta'])
        # convert beta (body incidence angle) to alpha (rotor tilt angle)
        self.add_subsystem('beta2alpha', om.ExecComp('alpha = arccos(sin_beta)', shape=(n_UAVs), alpha={'units' : 'rad'}), promotes_inputs=[('sin_beta', 'body|sin_beta')], promotes_outputs=[('alpha', 'rotor|alpha')])

        # thrust required by each rotor
        self.add_subsystem('thrust_each', ThrustOfEachRotor(n_UAVs=n_UAVs, n_rotor=n_rotor), promotes_inputs=['*'], promotes_outputs=['rotor|thrust'])

        # determine rotor revolution and thrust coeff given the advance ratio
        self.add_subsystem('rotor_revolution', RotorRevolutionFromAdvanceRatio(n_pts=n_UAVs), promotes_inputs=['UAVs|*', 'rotor|*', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        self.add_subsystem('CT', ThrustCoefficient(n_pts=n_UAVs, rho_air=rho_air), promotes_inputs=['*'], promotes_outputs=['*'])
        # set default advance ratio
        self.set_input_defaults('rotor|mu', 0.15 * np.ones(n_UAVs))

        # profile power
        self.add_subsystem('profile_power', ProfilePower(n_pts=n_UAVs, rho_air=rho_air, sigma=self.options['rotor_sigma']), promotes_inputs=['*'], promotes_outputs=['*'])
        # induced power
        self.add_subsystem('rotor_inflow', RotorInflow(n_pts=n_UAVs), promotes_inputs=['*'], promotes_outputs=['*'])   # this is an implicit component
        self.add_subsystem('v_induced', InducedVelocity(n_pts=n_UAVs), promotes_inputs=['UAVs|*', 'rotor|*', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        self.add_subsystem('kappa', InducedPowerFactor(n_pts=n_UAVs, FM=self.options['hover_FM'], rho_air=rho_air), promotes_inputs=['*'], promotes_outputs=['*'])

        # total power required in forward flight
        self.add_subsystem('power_req', PowerForwardComp(n_UAVs=n_UAVs, n_rotor=n_rotor), promotes_inputs=['rotor|*', 'v_induced', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        
        # add solvers for the implicit component.
        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50, iprint=2, rtol=1e-10)
        # self.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, iprint=2, rtol=1e-8)
        # self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        # self.nonlinear_solver.linesearch.options['maxiter'] = 10
        # self.nonlinear_solver.linesearch.options['iprint'] = 0
        # self.linear_solver = om.DirectSolver(assemble_jac=True)


class PowerForwardWithWing(om.Group):
    """
    Power in winged cruise (of Lift+cruise UAV)
    Inputs: UAV weight, wing area, cruise speed, rotor radius, prop advance ratio J
        rotor|alpha (rotor tilt angle w.r.t. vertical direction)

    Outputs: power (sum of all rotors), thrust of each rotor
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of cruising rotors')
        self.options.declare('hover_FM', types=float, desc='hover figure of merit')
        self.options.declare('rho_air', default=1.225, desc='air density')
        self.options.declare('Cd0', types=float, desc='minimum Cd of the drag polar')
        self.options.declare('wing_AR', types=float, desc='wing aspect ratio')
        self.options.declare('wing_e', types=float, desc='Oswald efficiency')
        self.options.declare('rotor_sigma', types=float, desc='rotor solidity')
        
    def setup(self):
        n_UAVs = self.options['n_UAVs']
        n_rotor = self.options['n_rotor']
        rho_air = self.options['rho_air']

        # lift required (= weight)
        lift_comp = om.ExecComp('lift = 9.81 * weight', shape=(n_UAVs,), lift={'units' : 'N'}, weight={'units' : 'kg'})
        self.add_subsystem('lift', lift_comp, promotes_inputs=[('weight', 'UAVs|W_total')], promotes_outputs=['*'])

        # drag in cruise.
        self.add_subsystem('drag', WingedCruiseDrag(n_UAVs=n_UAVs, rho_air=rho_air, Cd0=self.options['Cd0'], wing_AR=self.options['wing_AR'], wing_e=self.options['wing_e']), promotes_inputs=['*'], promotes_outputs=['drag', 'CL_cruise'])

        # thrust required by each rotor
        self.add_subsystem('thrust_each', ThrustOfEachRotor(n_UAVs=n_UAVs, n_rotor=n_rotor), promotes_inputs=[('thrust_all', 'drag')], promotes_outputs=['rotor|thrust'])

        # compute the rotor revolution given the propeller advance ratio J
        self.add_subsystem('prop_revolution', PropellerRevolutionFromAdvanceRatio(n_pts=n_UAVs), promotes_inputs=['UAVs|*', 'rotor|*', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        # set default propeller advance ratio
        self.set_input_defaults('rotor|J', 1.0 * np.ones(n_UAVs))

        # compute the rotor advance ratio mu (different from J!), and thrust coefficient
        self.add_subsystem('mu', RotorAdvanceRatio(n_pts=n_UAVs), promotes_inputs=['UAVs|*', 'rotor|*', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        self.add_subsystem('CT', ThrustCoefficient(n_pts=n_UAVs, rho_air=rho_air), promotes_inputs=['*'], promotes_outputs=['*'])

        # profile power
        self.add_subsystem('profile_power', ProfilePower(n_pts=n_UAVs, rho_air=rho_air, sigma=self.options['rotor_sigma']), promotes_inputs=['*'], promotes_outputs=['*'])
        # induced power
        self.add_subsystem('rotor_inflow', RotorInflow(n_pts=n_UAVs), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('v_induced', InducedVelocity(n_pts=n_UAVs), promotes_inputs=['UAVs|*', 'rotor|*', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        self.add_subsystem('kappa', InducedPowerFactor(n_pts=n_UAVs, FM=self.options['hover_FM'], rho_air=rho_air), promotes_inputs=['*'], promotes_outputs=['*'])
        
        # total power required in forward flight
        self.add_subsystem('power_req', PowerForwardComp(n_UAVs=n_UAVs, n_rotor=n_rotor), promotes_inputs=['rotor|*', 'v_induced', ('v_inf', 'UAVs|speed')], promotes_outputs=['*'])
        self.set_input_defaults('rotor|alpha', val=85 * np.ones(n_UAVs), units='deg')   # assume the rotor tilt angle to 85 deg (i.e. AoA = 5deg) in cruise, this will be used for mu computation etc.


# ------------------------------
# other components for power estimation
# ------------------------------
        
class PowerForwardComp(om.ExplicitComponent):
    # computes the power required in forward flight

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of rotor')

    def setup(self):
        n_pts = self.options['n_UAVs']

        self.add_input('rotor|thrust', shape=(n_pts,), units='N', desc='thrust of a rotor')
        self.add_input('rotor|alpha', shape=(n_pts,), units='rad', desc='rotor tilt angle.')  # alpha=90 when rotor is used as a propeller of aircraft. alpha=0 in hover.
        self.add_input('v_inf', shape=(n_pts,), units='m/s', desc='freestream velocity')
        self.add_input('v_induced', shape=(n_pts,), units='m/s', desc='induced velocity')
        self.add_input('rotor|profile_power', shape=(n_pts,), units='W', desc='profile power of a rotor, P0')
        self.add_input('rotor|kappa', shape=(n_pts,), desc='induced power factor')
        self.add_output('power_forward', shape=(n_pts,), units='W', desc='power required in forward flight (sum of all rotors)')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        n_rotor = self.options['n_rotor']
        speed_kappa = inputs['rotor|kappa'] * inputs['v_induced'] + inputs['v_inf'] * np.sin(inputs['rotor|alpha'])
        power_each = inputs['rotor|thrust'] * speed_kappa + inputs['rotor|profile_power']
        outputs['power_forward'] = n_rotor * power_each

    def compute_partials(self, inputs, partials):
        n_rotor = self.options['n_rotor']
        speed_kappa = inputs['rotor|kappa'] * inputs['v_induced'] + inputs['v_inf'] * np.sin(inputs['rotor|alpha'])

        partials['power_forward', 'rotor|thrust'] = n_rotor * speed_kappa
        partials['power_forward', 'rotor|alpha'] = n_rotor * (inputs['rotor|thrust'] * inputs['v_inf'] * np.cos(inputs['rotor|alpha']))
        partials['power_forward', 'v_inf'] = n_rotor * (inputs['rotor|thrust'] * np.sin(inputs['rotor|alpha']))
        partials['power_forward', 'v_induced'] = n_rotor * (inputs['rotor|kappa'] * inputs['rotor|thrust'])
        partials['power_forward', 'rotor|profile_power'] = n_rotor
        partials['power_forward', 'rotor|kappa'] = n_rotor * (inputs['rotor|thrust'] * inputs['v_induced'])


class ProfilePower(om.ExplicitComponent):
    """
    computes the profile power of the rotor
    Inputs: rotor radius, edgewise advance ratio, rotor angular velocity
    Outputs: profile power
    """

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')
        self.options.declare('rho_air', default=float, desc='air density')
        self.options.declare('sigma', types=float, desc='rotor solidity, e.g. 0.13')
        self.options.declare('cd0', default=0.012, desc='zero lift drag of rotor airfoil')
    
    def setup(self):
        n_pts = self.options['n_pts']

        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('rotor|mu', shape=(n_pts,), desc='advance ratio')
        self.add_input('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.add_output('rotor|profile_power', shape=(n_pts,), units='W', desc='profile power of one rotor, P0')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        rho_air = self.options['rho_air']
        sigma = self.options['sigma']
        cd0 = self.options['cd0']

        p0_each = sigma * cd0 / 8 * (1 + 4.65 * inputs['rotor|mu']**2) * (np.pi * rho_air * inputs['rotor|omega']**3 * inputs['UAVs|rotor_radius']**5)
        outputs['rotor|profile_power'] = p0_each

    def compute_partials(self, inputs, partials):
        rho_air = self.options['rho_air']
        sigma = self.options['sigma']
        cd0 = self.options['cd0']

        k1 = sigma * cd0 / 8
        k2 = (1 + 4.65 * inputs['rotor|mu']**2)
        k3 = (np.pi * rho_air * inputs['rotor|omega']**3 * inputs['UAVs|rotor_radius']**5)
        # p0_each = k1 * k2 * k3
        
        partials['rotor|profile_power', 'rotor|mu'] = k1 * k3 * (2 * 4.65 * inputs['rotor|mu'])
        partials['rotor|profile_power', 'UAVs|rotor_radius'] = k1 * k2 * (np.pi * rho_air * inputs['rotor|omega']**3 * 5 * inputs['UAVs|rotor_radius']**4)
        partials['rotor|profile_power', 'rotor|omega'] = k1 * k2 * (np.pi * rho_air * 3 * inputs['rotor|omega']**2 * inputs['UAVs|rotor_radius']**5)

# --------------------------------------------
# induced speed and induced power factor
# --------------------------------------------

class RotorInflow(om.ImplicitComponent):
    # computes the inflow of a rotor (lambda)

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')

    def setup(self):
        n_pts = self.options['n_pts']

        self.add_input('rotor|Ct', shape=(n_pts,), desc='thrust coefficient')
        self.add_input('rotor|mu', shape=(n_pts,), desc='advance ratio')
        self.add_input('rotor|alpha', shape=(n_pts,), units='rad', desc='rotor tilt angle.')  # alpha=90 when rotor is used as a propeller of aircraft. alpha=0 in hover.
        self.add_output('rotor|lambda', val=0.1 * np.ones(n_pts), desc='rotor inflow', lower=0., upper=10.)
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def apply_nonlinear(self, inputs, outputs, residuals):
        # compute residuals
        Ct = inputs['rotor|Ct']
        mu = inputs['rotor|mu']
        lamb = outputs['rotor|lambda']
        tan_a = np.tan(inputs['rotor|alpha'])
        residuals['rotor|lambda'] = mu * tan_a + Ct / (2 * np.sqrt(mu**2 + lamb**2)) - lamb

    def linearize(self, inputs, outputs, partials):
        Ct = inputs['rotor|Ct']
        mu = inputs['rotor|mu']
        lamb = outputs['rotor|lambda']
        tan_a = np.tan(inputs['rotor|alpha'])

        partials['rotor|lambda', 'rotor|Ct'] = 1 / (2 * np.sqrt(mu**2 + lamb**2))
        partials['rotor|lambda', 'rotor|mu'] = tan_a + Ct / 2 * (-mu * (mu**2 + lamb**2)**(-1.5))
        partials['rotor|lambda', 'rotor|alpha'] = mu / np.cos(inputs['rotor|alpha'])**2
        partials['rotor|lambda', 'rotor|lambda'] = Ct / 2 * (-lamb * (mu**2 + lamb**2)**(-1.5)) - 1


class InducedVelocity(om.ExplicitComponent):
    # computes the induced velocity

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')

    def setup(self):
        n_pts = self.options['n_pts']

        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.add_input('rotor|alpha', shape=(n_pts,), units='rad', desc='rotor tilt angle.')  # alpha=90 when rotor is used as a propeller of aircraft. alpha=0 in hover.
        self.add_input('rotor|lambda', shape=(n_pts,), desc='rotor inflow')
        self.add_input('v_inf', shape=(n_pts,), units='m/s', desc='freestream velocity')
        self.add_output('v_induced', shape=(n_pts,), units='m/s', desc='induced velocity')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        outputs['v_induced'] = inputs['rotor|omega'] * inputs['UAVs|rotor_radius'] * inputs['rotor|lambda'] - inputs['v_inf'] * np.sin(inputs['rotor|alpha'])

    def compute_partials(self, inputs, partials):
        partials['v_induced', 'UAVs|rotor_radius'] = inputs['rotor|omega'] * inputs['rotor|lambda']
        partials['v_induced', 'rotor|omega'] = inputs['UAVs|rotor_radius'] * inputs['rotor|lambda']
        partials['v_induced', 'rotor|lambda'] = inputs['rotor|omega'] * inputs['UAVs|rotor_radius']
        partials['v_induced', 'v_inf'] = -np.sin(inputs['rotor|alpha'])
        partials['v_induced', 'rotor|alpha'] = -inputs['v_inf'] * np.cos(inputs['rotor|alpha'])

class InducedPowerFactor(om.Group):
    """
    computes the induced power factor kappa in forward flight
    Inputs: rotor|thrust, UAVs|rotor_radius, rotor|profile_power
    Output: rotor|kappa
    """

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')
        self.options.declare('FM', types=float, desc='hover figure of merit')
        self.options.declare('rho_air', default=float, desc='air density')

    def setup(self):
        n_pts = self.options['n_pts']
        fm = self.options['FM']
        rho_air = self.options['rho_air']

        # compute kappa value
        self.add_subsystem('kappa_raw', InducedPowerFactorComp(n_pts=n_pts, FM=fm, rho_air=rho_air), promotes_inputs=['*'])
        # minimum value of kappa
        indep = self.add_subsystem('kappa_min', om.IndepVarComp())
        indep.add_output('kappa_min', val=1.15 * np.ones(n_pts))

        # soft max of (1.15, kappa_raw)
        self.add_subsystem('softmax', SoftMax(n_pts=n_pts, rho=30), promotes_outputs=[('fmax', 'rotor|kappa')])
        self.connect('kappa_raw.kappa_raw', 'softmax.f1')
        self.connect('kappa_min.kappa_min', 'softmax.f2')


class InducedPowerFactorComp(om.ExplicitComponent):
    # computes the induced power factor kappa in forward flight
    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')
        self.options.declare('FM', types=float, desc='hover figure-of-merit')
        self.options.declare('rho_air', default=float, desc='air density')

    def setup(self):
        n_pts = self.options['n_pts']

        self.add_input('rotor|thrust', shape=(n_pts,), units='N', desc='thrust of a rotor')
        self.add_input('UAVs|rotor_radius', val=0.3 * np.ones(n_pts), units='m', desc='rotor radius')
        self.add_input('rotor|profile_power', shape=(n_pts,), units='W', desc='profile power of a rotor, P0')
        self.add_output('kappa_raw', shape=(n_pts,), desc='induced power factor before imposing the minimum value')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        efficiency = self.options['FM']
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        area = np.pi * inputs['UAVs|rotor_radius']**2
        kappa = 1 / efficiency - np.sqrt(2 * rho_air * area) / thrust**1.5 * inputs['rotor|profile_power']
        outputs['kappa_raw'] = kappa

    def compute_partials(self, inputs, partials):
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        r = inputs['UAVs|rotor_radius']
        area = np.pi * inputs['UAVs|rotor_radius']**2
        # kappa = 1 / self.efficiency - np.sqrt(2 * self.rho_air * area) / thrust**1.5 * inputs['rotor|profile_power']

        dk_dr = -rho_air * (2 * rho_air * area)**-0.5 / thrust**1.5 * inputs['rotor|profile_power'] * 2 * np.pi * r
        dk_dt = 1.5 * np.sqrt(2 * rho_air * area) / thrust**2.5 * inputs['rotor|profile_power']
        dk_dp = - np.sqrt(2 * rho_air * area) / thrust**1.5

        partials['kappa_raw', 'rotor|thrust'] = dk_dt
        partials['kappa_raw', 'UAVs|rotor_radius'] = dk_dr
        partials['kappa_raw', 'rotor|profile_power'] = dk_dp


# --------------------------------------
#  advance ratio, thrust coeff, rotor revolution
# --------------------------------------

class ThrustCoefficient(om.ExplicitComponent):
    # computes the thrust coefficient
    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')
        self.options.declare('rho_air', default=float, desc='air density')

    def setup(self):
        n_pts = self.options['n_pts']
        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('rotor|thrust', shape=(n_pts,), units='N', desc='thrust of a rotor')
        self.add_input('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.add_output('rotor|Ct', shape=(n_pts,), desc='thrust coefficient')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        omega = inputs['rotor|omega']
        r = inputs['UAVs|rotor_radius']
        outputs['rotor|Ct'] = thrust / (np.pi * rho_air * omega**2 * r**4)

    def compute_partials(self, inputs, partials):
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        omega = inputs['rotor|omega']
        r = inputs['UAVs|rotor_radius']
        partials['rotor|Ct', 'UAVs|rotor_radius'] = thrust / (np.pi * rho_air * omega**2) * (-4 / r**5)
        partials['rotor|Ct', 'rotor|omega'] = thrust / (np.pi * rho_air * r**4) * (-2 / omega**3)
        partials['rotor|Ct', 'rotor|thrust'] = 1 / (np.pi * rho_air * omega**2 * r**4)

class RotorAdvanceRatio(om.ExplicitComponent):
    # computes the rotor advance ratio mu = V cos(alpha) / (omega * r)
    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')

    def setup(self):
        n_pts = self.options['n_pts']
        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('v_inf', shape=(n_pts,), units='m/s', desc='freestream velocity')
        self.add_input('rotor|alpha', shape=(n_pts,), units='rad', desc='rotor tilt angle.')
        self.add_input('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.add_output('rotor|mu', shape=(n_pts,), desc='advance ratio of rotor')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        v_inf = inputs['v_inf']
        alpha = inputs['rotor|alpha']
        r = inputs['UAVs|rotor_radius']
        omega = inputs['rotor|omega']
        outputs['rotor|mu'] = v_inf * np.cos(alpha) / r / omega

    def compute_partials(self, inputs, partials):
        v_inf = inputs['v_inf']
        alpha = inputs['rotor|alpha']
        r = inputs['UAVs|rotor_radius']
        omega = inputs['rotor|omega']
        partials['rotor|mu', 'UAVs|rotor_radius'] = -v_inf * np.cos(alpha) / omega / r**2
        partials['rotor|mu', 'v_inf'] = np.cos(alpha) / omega / r
        partials['rotor|mu', 'rotor|alpha'] = - v_inf * np.sin(alpha) / omega / r
        partials['rotor|mu', 'rotor|omega'] = -v_inf * np.cos(alpha) / r / omega**2

class RotorRevolutionFromCT(om.ExplicitComponent):
    # computes the rotor revolution (omega) given the thrust coefficient CT
    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')
        self.options.declare('rho_air', default=float, desc='air density')

    def setup(self):
        n_pts = self.options['n_pts']
        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('rotor|thrust', shape=(n_pts,), units='N', desc='thrust of a rotor')
        self.add_input('rotor|Ct', shape=(n_pts,), desc='thrust coefficient')
        self.add_output('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        r = inputs['UAVs|rotor_radius']
        Ct = inputs['rotor|Ct']
        outputs['rotor|omega'] = np.sqrt(thrust / (np.pi * rho_air * Ct * r**4))

    def compute_partials(self, inputs, partials):
        rho_air = self.options['rho_air']
        thrust = inputs['rotor|thrust']
        r = inputs['UAVs|rotor_radius']
        Ct = inputs['rotor|Ct']
        omega2 = thrust / (np.pi * self.rho_air * Ct * r**4)   # omega square
        domega2_domega = 0.5 / omega2

        partials['rotor|omega', 'UAVs|rotor_radius'] = thrust / (np.pi * rho_air * Ct) * (-4 / r**5) * domega2_domega
        partials['rotor|omega', 'rotor|Ct'] = thrust / (np.pi * rho_air * r**4) * (-1 / Ct**2) * domega2_domega
        partials['rotor|omega', 'rotor|thrust'] = 1 / (np.pi * rho_air * Ct * r**4) * domega2_domega


class RotorRevolutionFromAdvanceRatio(om.ExplicitComponent):
    # computes the rotor revolution (omega) given the avance ratio
    # The definition of advance ratio for rotor: mu = V cos(alpha) / (omega * r), where V cos(alpha) is the airspeed component parallel to the disk.
    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')

    def setup(self):
        n_pts = self.options['n_pts']
        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('v_inf', shape=(n_pts,), units='m/s', desc='freestream velocity')
        self.add_input('rotor|alpha', shape=(n_pts,), units='rad', desc='rotor tilt angle.')
        self.add_input('rotor|mu', shape=(n_pts,), desc='advance ratio of rotor')   # mu = V cos(alpha) / omega / r, where V cos(alpha) is the airspeed component parallel to the disk.
        self.add_output('rotor|omega', shape=(n_pts,), units='rad/s', desc='rotor angular velocity')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        v_inf = inputs['v_inf']
        alpha = inputs['rotor|alpha']
        r = inputs['UAVs|rotor_radius']
        mu = inputs['rotor|mu']
        outputs['rotor|omega'] = v_inf * np.cos(alpha) / mu / r

    def compute_partials(self, inputs, partials):
        v_inf = inputs['v_inf']
        alpha = inputs['rotor|alpha']
        r = inputs['UAVs|rotor_radius']
        mu = inputs['rotor|mu']
        partials['rotor|omega', 'UAVs|rotor_radius'] = -v_inf * np.cos(alpha) / mu / r**2
        partials['rotor|omega', 'v_inf'] = np.cos(alpha) / mu / r
        partials['rotor|omega', 'rotor|alpha'] = - v_inf * np.sin(alpha) / mu / r
        partials['rotor|omega', 'rotor|mu'] = -v_inf * np.cos(alpha) / r / mu**2


class PropellerRevolutionFromAdvanceRatio(om.ExplicitComponent):
    # computes the propeller revolution (omega) given the avance ratio.
    # The definition of advance ratio for propeller: J = V / (n D)

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='number of points (typically number of UAVs)')

    def setup(self):
        n_pts = self.options['n_pts']
        self.add_input('UAVs|rotor_radius', shape=(n_pts,), units='m', desc='rotor radius')
        self.add_input('v_inf', shape=(n_pts,), units='m/s', desc='freestream velocity')
        self.add_input('rotor|J', shape=(n_pts,), desc='advance ratio of propeller')
        self.add_output('rotor|omega', shape=(n_pts,), units='rad/s', desc='propeller angular velocity')
        self.declare_partials('*', '*', rows=np.arange(n_pts), cols=np.arange(n_pts))

    def compute(self, inputs, outputs):
        v_inf = inputs['v_inf']
        r = inputs['UAVs|rotor_radius']
        J = inputs['rotor|J']
        n = v_inf / (2 * r) / J   # revolutions-per-second
        outputs['rotor|omega'] = n * 2 * np.pi

    def compute_partials(self, inputs, partials):
        v_inf = inputs['v_inf']
        r = inputs['UAVs|rotor_radius']
        J = inputs['rotor|J']

        partials['rotor|omega', 'UAVs|rotor_radius'] = -2 * np.pi * v_inf / 2 / J / r**2
        partials['rotor|omega', 'v_inf'] = 2 * np.pi / (2 * r) / J
        partials['rotor|omega', 'rotor|J'] = -2 * np.pi * v_inf / (2 * r) / J**2


# ------------------------------
# trim components
# ------------------------------

class MultiRotorTrim(om.ExplicitComponent):
    """
    computes the body tilt angle for (wingless) multirotor in cruise
    Inputs: UAV weight, drag force
    Outputs: thrust required for trim, body tilt angle
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')

    def setup(self):
        n_UAVs = self.options['n_UAVs']

        self.add_input('UAVs|W_total', shape=(n_UAVs,), units='kg', desc='vehicle weight')
        self.add_input('drag', shape=(n_UAVs,), units='N', desc='drag')
        self.add_output('thrust', shape=(n_UAVs,), units='N', desc='thrust required (as a vehicle)')
        self.add_output('body|sin_beta', shape=(n_UAVs,), desc='sin(beta). beta = incidence angle of the body')
        self.declare_partials('*', '*', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs))

        self.gravity = 9.81

    def compute(self, inputs, outputs):
        weight = inputs['UAVs|W_total'] * self.gravity
        drag = inputs['drag']
        thrust = (weight**2 + drag**2)**0.5
        outputs['thrust'] = thrust
        outputs['body|sin_beta'] = weight / thrust

    def compute_partials(self, inputs, partials):
        weight = inputs['UAVs|W_total'] * self.gravity
        drag = inputs['drag']
        dw_dmass = self.gravity
        thrust = (weight**2 + drag**2)**0.5
        dt_dw = (0.5 / thrust) * 2 * weight * dw_dmass
        dt_dd = (0.5 / thrust) * 2 * drag
        dsinbeta_dt = - weight / thrust**2
        partials['thrust', 'drag'] = (0.5 / thrust) * 2 * drag
        partials['thrust', 'UAVs|W_total'] = (0.5 / thrust) * 2 * weight * dw_dmass
        partials['body|sin_beta', 'UAVs|W_total'] = dw_dmass / thrust + dsinbeta_dt * dt_dw
        partials['body|sin_beta', 'drag'] = dsinbeta_dt * dt_dd


class ThrustOfEachRotor(om.ExplicitComponent):
    # computes the thrust required by each rotor given the weight or drag requirement
    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs')
        self.options.declare('n_rotor', types=int, desc='number of rotor')

    def setup(self):
        n_UAVs = self.options['n_UAVs']
        n_rotor = self.options['n_rotor']

        self.add_input('thrust_all', shape=(n_UAVs,), units='N', desc='thrust required (sum of all rotors)')
        self.add_output('rotor|thrust', shape=(n_UAVs,), units='N', desc='thrust required by each rotor')
        # partial is constant
        self.declare_partials('rotor|thrust', 'thrust_all', rows=np.arange(n_UAVs), cols=np.arange(n_UAVs), val=np.ones(n_UAVs) / n_rotor)

    def compute(self, inputs, outputs):
        outputs['rotor|thrust'] = inputs['thrust_all'] / self.options['n_rotor']