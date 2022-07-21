"""
Runscript to estimate energy consumption for the given mission and UAV specifications.

In this example, I compare the energy consumptions by two UAV configurations: multirotor and Lift+Cruise.
The total weight is set to 5 kg for both UAVs. See the code below for the detailed settings of the UAV design parameters etc.
The mission consists of 240 sec of hover and 100~5000 m of cruise. Hovering time includes the times for vertical climb/descent (reasonable assumption as long as you climb/descent slowly)
Then, I compute the energy consumption of each UAV to fly the missions.
"""

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from evtolsizing.sizing_groups import EnergyConsumption

if __name__ == '__main__':
    # ======================================
    # --- define multirotor UAV parameters ---
    uav1_params = {}
    uav1_params['config'] = 'multirotor'
    # rotor parameters
    uav1_params['n_rotors_lift'] = 6   # number of lifting rotors
    uav1_params['rotor_lift_solidity'] = 0.13   # solidity of lifting rotor
    uav1_params['hover_FM'] = 0.75  # hover figure of merit
    # battery parameters
    uav1_params['battery_rho'] = 158.  # battery energy density, Wh / kg
    uav1_params['battery_eff'] = 0.85  # battery efficiency to consider power losses and avionics power
    uav1_params['battery_max_discharge'] = 0.7  # maximum battery discharge, something like 70%?
    # UAV designs
    uav1_weight = 5  # kg
    uav1_rotor_radius = 0.2   # lifting rotor radius, m
    uav1_speed = 10   # cruise speed, m/s

    # --- define Lift+Cruise UAV parameters ---
    uav2_params = {}
    uav2_params['config'] = 'lift+cruise'
    # rotor parameters
    uav2_params['n_rotors_lift'] = 4   # number of lifting rotors
    uav2_params['n_rotors_cruise'] = 2   # number of cruising rotors
    uav2_params['rotor_lift_solidity'] = 0.13   # solidity of lifting rotor
    uav2_params['rotor_cruise_solidity'] = 0.13   # solidity of cruising rotor
    uav2_params['hover_FM'] = 0.75  # hover figure of merit
    # wing parameters
    uav2_params['Cd0'] = 0.0397   # zero-lift drag coefficient. from Bacchini et al. 2021, "Impact of lift propeller drag on the performance of eVTOL lift+cruise aircraft", Aerospace Science and Technology
    uav2_params['wing_AR'] = 6.0  # wing aspect ratio
    uav2_params['wing_e'] = 0.8  # Oswald efficiency
    # battery parameters
    uav2_params['battery_rho'] = 158.  # battery energy density, Wh / kg
    uav2_params['battery_eff'] = 0.85  # battery efficiency to consider power losses and avionics power
    uav2_params['battery_max_discharge'] = 0.7  # maximum battery discharge, something like 70%?
    # UAV designs
    uav2_weight = 5  # kg
    uav2_rotor_radius_lift = 0.2   # lifting rotor radius, m
    uav2_rotor_radius_cruise = 0.2   # cruising rotor radius, m
    uav2_speed = 20   # cruise speed, m/s
    uav2_wing_area = 0.2  # m**2

    # --- define mission requirements ---
    # Here, we analyze 10 missions (of various ranges) at once.
    n_missions = 10
    flight_range = np.linspace(100, 5000, n_missions)  # total flight range, m
    hover_time = 240 * np.ones(n_missions)  # time to hover, s. vertical climb/descent should be included as hover (not a bad assumption if you climb/descent slowly)
    # ======================================

    # --- 1) energy consumption analysis for multirotor ---
    prob1 = om.Problem()
    # define model inputs
    indeps = prob1.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    # mission reqs
    indeps.add_output('flight_distance', flight_range, units='m')
    indeps.add_output('hover_time', hover_time, units='s')
    # design variables (and their initial guess)
    indeps.add_output('UAVs|W_total', uav1_weight * np.ones(n_missions), units='kg')   # total weight including payload
    indeps.add_output('UAVs|speed', uav1_speed * np.ones(n_missions), units='m/s')   # cruise speed
    indeps.add_output('UAVs|rotor_radius_lift', uav1_rotor_radius * np.ones(n_missions), units='m')   # lifting rotor radius
    indeps.add_output('rotor|mu', 0.3 * np.ones(n_missions), units=None)  # edgewise advance ratio in cruise

    # add energy consumption model
    prob1.model.add_subsystem('energy_model', EnergyConsumption(n_UAVs=n_missions, UAV_options=uav1_params), promotes_inputs=['*'], promotes_outputs=['*'])

    # run analysis and get results
    prob1.setup(check=False)
    prob1.run_model()
    energy1 = prob1.get_val('energy_cnsmp', 'W*h')   # Wh

    # --- 2) energy consumption analysis for Lift+Cruise ---
    prob2 = om.Problem()
    # define model inputs
    indeps = prob2.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    # mission reqs
    indeps.add_output('flight_distance', flight_range, units='m')
    indeps.add_output('hover_time', hover_time, units='s')
    # design variables (and their initial guess)
    indeps.add_output('UAVs|W_total', uav2_weight * np.ones(n_missions), units='kg')   # total weight including payload
    indeps.add_output('UAVs|speed', uav2_speed * np.ones(n_missions), units='m/s')   # cruise speed
    indeps.add_output('UAVs|rotor_radius_lift', uav2_rotor_radius_lift * np.ones(n_missions), units='m')   # lifting rotor radius
    indeps.add_output('UAVs|rotor_radius_cruise', uav2_rotor_radius_cruise * np.ones(n_missions), units='m')   # cruising rotor radius
    indeps.add_output('UAVs|S_wing', uav2_wing_area * np.ones(n_missions), units='m**2')  # wing area
    indeps.add_output('rotor|J', 1.0 * np.ones(n_missions), units=None)  # propeller advance ratio

    # add energy consumption model
    prob2.model.add_subsystem('energy_model', EnergyConsumption(n_UAVs=n_missions, UAV_options=uav2_params), promotes_inputs=['*'], promotes_outputs=['*'])

    # run analysis and get results
    prob2.setup(check=False)
    prob2.run_model()
    energy2 = prob2.get_val('energy_cnsmp', 'W*h')   # Wh

    # --- plot results ---
    # plot the mission range vs energy requirement
    plt.figure()
    plt.plot(flight_range / 1000, energy1, label='Multirotor')
    plt.plot(flight_range / 1000, energy2, label='Lift+Cruise')
    plt.legend()
    plt.xlabel('Flight range, km')
    plt.ylabel('Energy consumption, Wh')
    plt.show()
