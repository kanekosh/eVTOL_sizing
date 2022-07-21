"""
Runscript to minimize the UAV weight for given mission requirement.
Lift+cruise configuration.

In this example, I will find a minimum-weight Lift+Cruise UAV design. I assume the UAV has 4 lifting rotors and 2 cruising rotors.
I require the UAV to carry a 2 kg payload, fly 5000 m, and hover for 240 sec. Hovering time includes the times for vertical clibm and descent (if you climb/descent slowly, you can approximate climb/descent as hover).
Optimization variables are: UAV total weight (including the payload), wing area, lifting rotor radius, cruising rotor radius, cruise speed, and propeller advance ratio in cruise.
"""

import numpy as np
import openmdao.api as om
from evtolsizing.sizing_groups import WeightEstimation
from evtolsizing.utils import AddScalarVectorComp

if __name__ == '__main__':
    # ======================================
    # define UAV parameters
    uav_params = {}
    uav_params['config'] = 'lift+cruise'
    # rotor parameters
    uav_params['n_rotors_lift'] = 4   # number of lifting rotors
    uav_params['n_rotors_cruise'] = 2   # number of cruising rotors
    uav_params['rotor_lift_solidity'] = 0.13   # solidity of lifting rotor
    uav_params['rotor_cruise_solidity'] = 0.13   # solidity of cruising rotor
    uav_params['hover_FM'] = 0.75  # hover figure of merit
    # wing parameters
    uav_params['Cd0'] = 0.0397   # zero-lift drag coefficient. from Bacchini et al. 2021, "Impact of lift propeller drag on the performance of eVTOL lift+cruise aircraft", Aerospace Science and Technology
    uav_params['wing_AR'] = 6.0  # wing aspect ratio
    uav_params['wing_e'] = 0.8  # Oswald efficiency
    # battery parameters
    uav_params['battery_rho'] = 158.  # battery energy density, Wh / kg
    uav_params['battery_eff'] = 0.85  # battery efficiency to consider power losses and avionics power
    uav_params['battery_max_discharge'] = 0.7  # maximum battery discharge, something like 70%?

    # define mission requirements
    payload_weight = 2  # kg
    flight_range = 5000  # total flight range, m
    hover_time = 240  # time to hover, s. vertical climb/descent should be included as hover (not a bad assumption if you climb/descent slowly)
    # ======================================

    # --- setup OpenMDAO problem ----
    prob = om.Problem()

    # define model inputs
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    # mission reqs
    indeps.add_output('payload_weight', payload_weight, units='kg')
    indeps.add_output('flight_distance', flight_range, units='m')
    indeps.add_output('hover_time', hover_time, units='s')
    # design variables (and their initial guess)
    indeps.add_output('UAVs|W_total', 10., units='kg')   # total weight including payload
    indeps.add_output('UAVs|speed', 25., units='m/s')   # cruise speed
    indeps.add_output('UAVs|rotor_radius_lift', 0.2, units='m')   # lifting rotor radius
    indeps.add_output('UAVs|rotor_radius_cruise', 0.2, units='m')   # cruising rotor radius
    indeps.add_output('rotor|J', 1.0, units=None)  # prop advance ratio
    indeps.add_output('UAVs|S_wing', 0.2, units='m**2')   # wing area

    # add UAV weight estimation model
    n_UAVs = 1  # number of UAVs to be designed simutaneously
    prob.model.add_subsystem('uav_weight_model', WeightEstimation(n_UAVs=n_UAVs, UAV_options=uav_params), promotes_inputs=['*'], promotes_outputs=['*'])

    # sum total weight of UAVs (to be minimized). Not relevant if designing only one UAV, i.e., n_UAVs=1
    prob.model.add_subsystem('sum_mass', AddScalarVectorComp(vec_dim=n_UAVs, scalar_scale=0., vector_scale=np.ones(n_UAVs), units='kg'), promotes_inputs=[('vector_input', 'UAVs|W_total')], promotes_outputs=[('scalar_output', 'sum_W0')])

    # --- optimization problem ---
    # design variables and lower/upper bounds
    prob.model.add_design_var('UAVs|W_total', lower=0.1, upper=20, ref=10, units='kg')   # UAV total weight
    prob.model.add_design_var('UAVs|speed', lower=2., upper=30., ref=10, units='m/s')  # cruise speed
    prob.model.add_design_var('UAVs|rotor_radius_lift', lower=0.05, upper=0.25, ref=0.1, units='m')   # lifting rotor radius
    prob.model.add_design_var('UAVs|rotor_radius_cruise', lower=0.05, upper=0.25, ref=0.1, units='m')   # cruising rotor radius
    prob.model.add_design_var('rotor|J', lower=0.01, upper=1.3)  # edgewise advance ratio in cruise
    prob.model.add_design_var('UAVs|S_wing', lower=0.01, upper=1.0, units='m**2')
    # constraints
    prob.model.add_constraint('W_residual', lower=0.0, upper=0.0, ref=10)   # derive weight residual to 0
    prob.model.add_constraint('disk_loading_hover', upper=250.0, ref=100, units='N/m**2')   # limit disk loading
    prob.model.add_constraint('disk_loading_cruise', upper=250.0, ref=100, units='N/m**2')
    prob.model.add_constraint('rotor|Ct', lower=0.0, upper=0.14 * uav_params['rotor_cruise_solidity'], ref=0.01)   # in cruise. CT / solidity <= 0.14 to avoid too high blade loading
    prob.model.add_constraint('CL_cruise', lower=0.0, upper=0.6, ref=0.5)   # CL max at cruise = 0.6
    # objective: minimize mass
    prob.model.add_objective('sum_W0', ref=10)

    # optimizer settings
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-8
    prob.driver.options['disp'] = True

    # run optimization
    prob.setup(check=False)
    prob.run_driver()
    # prob.run_model()

    # prob.check_partials(compact_print=True)
    # om.n2(prob)

    # get weights
    W_total = prob.get_val('UAVs|W_total', 'kg')
    W_payload = prob.get_val('payload_weight', 'kg')
    W_battery = prob.get_val('UAVs|W_battery', 'kg')
    W_rotors = prob.get_val('UAVs|W_rotor_all', 'kg')
    W_motors = prob.get_val('UAVs|W_motor_all', 'kg')
    W_wing = prob.get_val('UAVs|W_wing', 'kg')
    W_frame = W_total - (W_payload + W_battery + W_rotors + W_motors + W_wing)

    # --- print results ---
    print('--------------------------------------------')
    print('--- problem settings ---')
    print('  UAV parameters echo:', uav_params)
    print('  payload weight [kg]:', list(W_payload))
    print('  flight range [m]   :', list(prob.get_val('flight_distance', 'm')))
    print('  hovering time [s]  :', list(prob.get_val('hover_time', 's')))
    print('\n--- design optimization results ---')
    print('Design variables')
    print('  lifting rotor radius [m] :', list(prob.get_val('UAVs|rotor_radius_lift', 'm')))
    print('  cruising rotor radius [m] :', list(prob.get_val('UAVs|rotor_radius_cruise', 'm')))
    print('  cruise speed [m/s]       :', list(prob.get_val('UAVs|speed', 'm/s')))
    print('  wing area [m**2] :', list(prob.get_val('UAVs|S_wing', 'm**2')))
    print('  prop advance ratio J:', list(prob.get_val('rotor|J')))
    print('Component weights [kg]')
    print('  total weight :', list(W_total))
    print('  payload      :', list(W_payload))
    print('  battery      :', list(W_battery))
    print('  rotors       :', list(W_rotors))
    print('  motors + ESCs:', list(W_motors))
    print('  wing         :', W_wing)
    print('  frame        :', W_frame)
    print('Performances')
    print('  power in hover: [W] :', list(prob.get_val('power_hover', 'W')))
    print('  power in cruise: [W]:', list(prob.get_val('power_forward', 'W')))
    print('  CL in cruise:', list(prob.get_val('CL_cruise')))
    print('Sanity check: W_residual [kg]:', list(prob.get_val('W_residual', 'kg')), ' = 0?')
    print('--------------------------------------------')
