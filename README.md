# eVTOL UAV sizing tool
eVTOL UAV sizing optimization tool using very simple models.
This tool is implemented using NASA's [OpenMDAO framework](https://openmdao.org)

## Installation
First, install Python 3.
I personally prefer `pyenv` and `virtualenv`, but any methods (e.g., `anaconda` or `miniconda`) should work.

Next, clone this repository by
```
$ git clone git@github.com:kanekosh/eVTOL_sizing.git
```
Then, navigate to this repository, and pip-install
```
$ cd eVTOL_sizing
$ pip install -e .
```
I recommend installing in the editable mode by specifying `-e` option if you'll be modifying the code.

## Run examples
Example scripts are available in `eVTOL_sizing/examples`.

Run the first example by
```
$ python estimate_energy_consumption.py
```
This example estimates the energy consumption of a UAV flight given mission requirements and UAV specifications.

Run the other examples by
```
$ python estimate_weight_multirotor.py
$ python estimate_weight_liftcruise.py
```
These examples optimize the UAV sizing variables given mission requirements.
Currently, two UAV configurations are available to analyze/design: multirotor and Lift+Cruise.

Please refer to the comments in the scripts for more details.

## References
* Chapter 3 of [my papar](http://websites.umich.edu/~mdolaboratory/pdf/Kaneko2022a.pdf)
* [This paper](https://arc.aiaa.org/doi/10.2514/1.C035805) discusses more details

## Notes
* This package is provided to the Michigan Vertical Flight Technology (MVFT) team without warranties or conditions of any kind: the models might be wrong, and the code might have bugs.
* Please feel free to share the code within the MVFT team, but please do not actively distribute it further (although this repo is public).

## Recommendations
Do not take the outputs of this package as is: the models used here are very simple and conceptual, and there is probably a big gap between my models and actual vehicles.
The main purpose of this code is to show an example of how to use optimization for UAV sizing and weight estimation, and hopefully to serve as a starting point for you.

I encourage you to modify the code/models to make them more practical for your use.
Forking the repo would be a good way to have your own version of the code.
For example, you may want to
* change the lower/upper bounds of design variables, e.g., rotor radius and cruise speed.
* update the battery energy density (currently 158 Wh/kg) once you decide which battery to equip.
* tune the value of the hover figure-of-merit based on experimental data.
* update the aerodynamic models using wind tunnel data.
* update the frame weight estimation based on test vehicles you build (currently I assume `frame_weight = 0.5 (kg) + 0.2 total_weight`, which is too simple).
* update the component weights (rotor, motor, ESC, etc) once you purchased the parts.