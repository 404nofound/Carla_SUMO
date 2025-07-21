# Carla_SUMO
**Still Updating**

Manual control or autonomous driving takeover experiment with CARLA-SUMO co-simulation

<img align="center" src="https://github.com/404nofound/Carla_SUMO/blob/main/media/exp_env.png" alt="" width="740" height="500" style="display: inline; float: right"/>

## Demo

**Still Updating**

## File Structure

`/roadrunner/`: map files used to generate Carla map and Sumo road network

`/Carla/PythonAPI/examples/`: Carla client script, added or replaced in your local files

`/Carla/Co-Simulation/Sumo/`: Sumo scripts, added or replaced in your local files

## Steps

1. Install UE4, Carla, Sumo, Roadrunner, etc. (follow documents: https://carla.readthedocs.io/en/latest/build_windows/; https://zhuanlan.zhihu.com/p/552983835)

2. Generate Carla map based on roadrunner `.xodr` file
3. Generate Sumo road network based on roadrunner `.xodr` file (follow `/Carla/Co-Simulation/Sumo/code.txt`)
4. Build `.exe` program in UE4 by `make package` in `x64 Native Tools Command Prompt for VS 2019` (follow https://zhuanlan.zhihu.com/p/552983835)

4. Run server program(`.exe`), execute Sumo file by `python run_synchronization.py examples/exp1.sumocfg --sumo-gui`, run Carla client script by `python lab_manual_control_part1.py`

