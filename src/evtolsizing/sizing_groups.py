import numpy as np
import openmdao.api as om

from evtolsizing.components_prop import PowerHover, PowerForwardEdgewise, PowerForwardWithWing
from evtolsizing.components_weight import PropulsionWeight, WingWeight

class EnergyConsumption(om.Group):
    """
    Computes the energy consumption of UAVs given the vehicle specifications and mission definisions.
    It also computes disk loading in hover and cruise (althrough disk loading is not relevant to energy consumption)

    Inputs:
    (mission requirements)
        flight_distance
        hover_time
    (UAV design variables)
        UAVs|W_total (total weight including the payload)
        UAVs|speed
        UAVs|rotor_radius_lift
        UAVs|rotor_radius_cruise (for lift_cruise only)
        UAVs|S_wing              (for lift_cruise only)
        rotor|mu                 (for multirotor only)
        rotor|J                  (for lift_cruise only)

    Outputs:
    (major performances)
        power_hover
        power_cruise
    (for some constraints)
        disk_loading_hover
        disk_loading_cruise
        rotor|Ct  (in cruise)
        CL_cruise (for QBiT)        
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs to be evaluated')
        self.options.declare('UAV_options', types=dict, desc='dict containing all option parameters')

    def setup(self):
        n_UAVs = self.options['n_UAVs']
        params = self.options['UAV_options']
        # unpack options
        config = params['config']
        n_rotors_lift = params['n_rotors_lift']  # number of lifting rotors
        rotor_lift_solidity = params['rotor_lift_solidity']  # solidity of lifting rotors
        hover_FM = params['hover_FM']  # figure of merit in hover
        if config == 'multirotor':
            pass
        elif config == 'lift+cruise':
            n_rotors_cruise = params['n_rotors_cruise']  # number of cruising rotors, only for lift+cruise
            Cd0 = params['Cd0']  # minimum drag of the drag polar
            wing_AR = params['wing_AR']  # wing aspect ratio
            wing_e = params['wing_e']  # Oswald efficiency
            rotor_cruise_solidity = params['rotor_cruise_solidity']  # solidity of cruising rotors
        else:
            raise RuntimeError('config must be hexa or qbit')

        # --- compute power consumptions for each flight segment ---
        # power in hover
        self.add_subsystem('power_hover', PowerHover(n_UAVs=n_UAVs, n_rotor=n_rotors_lift, hover_FM=hover_FM), promotes_inputs=['UAVs|W_total', ('UAVs|rotor_radius', 'UAVs|rotor_radius_lift')], promotes_outputs=['power_hover'])

        # power in cruise
        if config == 'lift+cruise':
            # UAV has wings.
            inputs_list = ['UAVs|W_total', 'UAVs|speed', 'UAVs|S_wing', 'rotor|J', ('UAVs|rotor_radius', 'UAVs|rotor_radius_cruise')]  # cruising rotor is used for cruise
            self.add_subsystem('power_forward_wing', PowerForwardWithWing(n_UAVs=n_UAVs, n_rotor=n_rotors_cruise, hover_FM=hover_FM, Cd0=Cd0, wing_AR=wing_AR, wing_e=wing_e, rotor_sigma=rotor_cruise_solidity), promotes_inputs=inputs_list, promotes_outputs=['*'])
        else:  # no wing
            inputs_list = ['UAVs|W_total', 'UAVs|speed', 'rotor|mu', ('UAVs|rotor_radius', 'UAVs|rotor_radius_lift')]   # lifting rotor is used for cruise
            self.add_subsystem('power_forward_edgewise', PowerForwardEdgewise(n_UAVs=n_UAVs, n_rotor=n_rotors_lift, hover_FM=hover_FM, rotor_sigma=rotor_lift_solidity), promotes_inputs=inputs_list , promotes_outputs=['*'])
        # END IF

        # --- compute energy consumption ---
        # energy = power_hover * hover_time + power_cruise * cruise_time
        energy_comp = om.ExecComp('energy_cnsmp = (power_hover * hover_time) + (power_forward * flight_distance / speed)',
                                    energy_cnsmp={'units' : 'W*s', 'shape' : (n_UAVs,)},
                                    power_hover={'units' : 'W', 'shape' : (n_UAVs,)},
                                    power_forward={'units' : 'W', 'shape' : (n_UAVs,)},
                                    hover_time={'units' : 's', 'shape' : (n_UAVs,)},
                                    flight_distance={'units' : 'm', 'shape' : (n_UAVs,)},
                                    speed={'units' : 'm/s', 'shape' : (n_UAVs,)})
        self.add_subsystem('energy', energy_comp, promotes_inputs=['power_hover', 'power_forward', 'hover_time', 'flight_distance', ('speed', 'UAVs|speed')], promotes_outputs=['energy_cnsmp'])

        # --- compute disk loadings ---
        # in hover
        disk_loading_comp_1 = om.ExecComp('disk_loading = thrust / pi / radius**2', disk_loading={'units' : 'N/m**2', 'shape' : (n_UAVs,)}, thrust={'units' : 'N', 'shape' : (n_UAVs,)}, radius={'units' : 'm', 'shape' : (n_UAVs,)})
        self.add_subsystem('disk_loading_hover', disk_loading_comp_1, promotes_inputs=[('radius', 'UAVs|rotor_radius_lift')], promotes_outputs=[('disk_loading', 'disk_loading_hover')])
        self.connect('power_hover.thrust_each', 'disk_loading_hover.thrust')
        # in cruise
        disk_loading_comp_2 = om.ExecComp('disk_loading = thrust / pi / radius**2', disk_loading={'units' : 'N/m**2', 'shape' : (n_UAVs,)}, thrust={'units' : 'N', 'shape' : (n_UAVs,)}, radius={'units' : 'm', 'shape' : (n_UAVs,)})
        self.add_subsystem('disk_loading_cruise', disk_loading_comp_2, promotes_outputs=[('disk_loading', 'disk_loading_cruise')])
        self.connect('rotor|thrust', 'disk_loading_cruise.thrust')
        if config == 'lift+cruise':
            self.promotes('disk_loading_cruise', inputs=[('radius', 'UAVs|rotor_radius_cruise')])
        else:  # multirotor
            self.promotes('disk_loading_cruise', inputs=[('radius', 'UAVs|rotor_radius_lift')])
        # END IF

        # --- add nonlinear solvers for implicit relations ---
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0, rtol=1e-10)
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['reraise_child_analysiserror'] = True
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 0
        self.linear_solver = om.DirectSolver(assemble_jac=True)


class WeightEstimation(om.Group):
    """
    UAV weight estimation given the design variables and mission requirement.
    Must be used with an optimizer to converge the weight residual.
    
    Inputs:
    (mission requirements)
        flight_distance
        hover_time
        payload_weight
    (UAV design variables)
        UAVs|W_total (total weight including the payload. This is an input, not output!)
        UAVs|speed
        UAVs|rotor_radius_lift
        UAVs|rotor_radius_cruise (for lift_cruise only)
        UAVs|S_wing              (for lift_cruise only)
        rotor|mu                 (for multirotor)
        rotor|J                  (for lift_cruise only)

    Outputs:
    (weight of each component)
        UAVs|W_battery
        UAVs|W_rotor_all
        UAVs|W_motor_all
    (major performances)
        power_hover
        power_cruise
    (for some constraints)
        disk_loading_hover
        disk_loading_cruise
        rotor|Ct  (in cruise)
        CL_cruise (for QBiT)        
    """

    def initialize(self):
        self.options.declare('n_UAVs', types=int, default=1, desc='number of UAVs to be evaluated')
        self.options.declare('UAV_options', types=dict, desc='dict containing all option parameters')

    def setup(self):
        n_UAVs = self.options['n_UAVs']
        params = self.options['UAV_options']
        # unpack options
        config = params['config']
        battery_rho = params['battery_rho']
        battery_eff = params['battery_eff']
        battery_max_discharge = params['battery_max_discharge']
        n_rotors_lift = params['n_rotors_lift']  # number of lifting rotors
        # hover_FM = params['hover_FM']  # figure of merit in hover
        if config == 'multirotor':
            pass
        elif config == 'lift+cruise':
            n_rotors_cruise = params['n_rotors_cruise']  # number of cruising rotors, only for lift+cruise
        else:
            raise RuntimeError('config must be hexa or qbit')

        # --- energy required to fly the given mission ---
        self.add_subsystem('energy', EnergyConsumption(n_UAVs=n_UAVs, UAV_options=params), promotes_inputs=['*'], promotes_outputs=['*'])

        # --- weight estimation of each component ---
        # battery weight
        # the required energy output from the battery is: energy_output = energy_cnsmp / battery_eff, considering loss and avionics power etc.
        # also, batteries usually cannot discharge all of its stored energy, so we need to consider that as well
        battery_weight_comp = om.ExecComp('W_battery = energy_req / (battery_rho * battery_eff * battery_max_discharge)',
                                            W_battery={'units' : 'kg', 'shape' : (n_UAVs,)},
                                            energy_req={'units' : 'W*h', 'shape' : (n_UAVs,)},
                                            battery_rho={'units' : 'W*h/kg', 'val' : battery_rho},
                                            battery_eff={'val' : battery_eff},
                                            battery_max_discharge={'val' : battery_max_discharge})
        self.add_subsystem('battery_weight', battery_weight_comp, promotes_inputs=[('energy_req', 'energy_cnsmp'), ], promotes_outputs=[('W_battery', 'UAVs|W_battery')])

        # propulsion weight
        # --- weight ---
        if config == 'multirotor':
            self.add_subsystem('propulsion_weight', PropulsionWeight(n_UAVs=n_UAVs, n_rotor=n_rotors_lift), promotes_inputs=[('UAVs|rotor_radius', 'UAVs|rotor_radius_lift')], promotes_outputs=['UAVs|W_rotor_all', 'UAVs|W_motor_all'])
            self.connect('power_hover', 'propulsion_weight.power_max')   # assume max power output = power in hover
        elif config == 'lift+cruise':
            # lifting rotors
            self.add_subsystem('propulsion_weight_lift', PropulsionWeight(n_UAVs=n_UAVs, n_rotor=n_rotors_lift), promotes_inputs=[('UAVs|rotor_radius', 'UAVs|rotor_radius_lift')], promotes_outputs=[('UAVs|W_rotor_all', 'W_rotors_lift'), ('UAVs|W_motor_all', 'W_motors_lift')])
            self.connect('power_hover', 'propulsion_weight_lift.power_max')   # assume max power output = power in hover
            # cruising rotors
            self.add_subsystem('propulsion_weight_cruise', PropulsionWeight(n_UAVs=n_UAVs, n_rotor=n_rotors_cruise), promotes_inputs=[('UAVs|rotor_radius', 'UAVs|rotor_radius_cruise')], promotes_outputs=[('UAVs|W_rotor_all', 'W_rotors_cruise'), ('UAVs|W_motor_all', 'W_motors_cruise')])
            self.connect('power_forward', 'propulsion_weight_cruise.power_max')   # assume max power output = power in cruise
            # sum both systems weight
            adder = om.AddSubtractComp()
            adder.add_equation('W_rotors', input_names=['W_rotors_lift', 'W_rotors_cruise'], vec_size=n_UAVs, units='kg', scaling_factors=[1., 1.])
            adder.add_equation('W_motors', input_names=['W_motors_lift', 'W_motors_cruise'], vec_size=n_UAVs, units='kg', scaling_factors=[1., 1.])
            self.add_subsystem('propulsion_weight', adder, promotes_inputs=['*'], promotes_outputs=[('W_rotors', 'UAVs|W_rotor_all'), ('W_motors', 'UAVs|W_motor_all')])
        # EBD IF

        # wing weight
        if config == 'lift+cruise':
            self.add_subsystem('wing_weight', WingWeight(n_UAVs=n_UAVs), promotes_inputs=['*'], promotes_outputs=['*'])
        # END IF

        # compute weight resigual
        prom_in = [('W_total', 'UAVs|W_total'), ('W_payload', 'payload_weight'), ('W_battery', 'UAVs|W_battery'), ('W_rotor_all', 'UAVs|W_rotor_all'), ('W_motor_all', 'UAVs|W_motor_all'), ('W_wing', 'UAVs|W_wing')]
        # W_residual needs to be driven to 0 by nonlinear solver or optimization constraint
        # frame weight is assumed to be: W_frame = 0.5 (kg) + 0.2 W_total
        w_residual_eqn = 'W_residual = W_total - W_payload - W_battery - W_rotor_all - W_motor_all - 0.2 * W_total - W_wing - 0.5'
        self.add_subsystem('W_residual', om.ExecComp(w_residual_eqn, shape=(n_UAVs,), units='kg'), promotes_inputs=prom_in, promotes_outputs=['W_residual'])
        if config == 'multirotor':
            # for wingless multirotor, set 0 weight for wing
            self.set_input_defaults('UAVs|W_wing', np.zeros(n_UAVs))
        # END IF

        """
        # use nonlinear solver to impose W_residual = 0 by varying W_total. set LB and UB of W_total here.
        residual_balance = om.BalanceComp('UAVs|W_total', units='kg', eq_units='kg', lhs_name='W_residual', rhs_name='zeros', lower=1.00, upper=50.0, val=20 * np.zeros(n_UAVs), rhs_val=np.zeros(n_UAVs), normalize=False)
        self.add_subsystem('weight_balance', residual_balance, promotes_inputs=['W_residual'], promotes_outputs=['UAVs|W_total'])
        self.set_input_defaults('weight_balance.zeros', np.zeros(n_UAVs))

        # add solvers for implicit relations
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0, rtol=1e-10)
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['reraise_child_analysiserror'] = True
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 0
        self.linear_solver = om.DirectSolver(assemble_jac=True)
        """
    