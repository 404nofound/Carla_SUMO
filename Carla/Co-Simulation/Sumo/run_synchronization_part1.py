#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# Modify: Yichang Shao
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""
import datetime
import winsound
# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time

# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import carla as CARLA
import glob
import os
import sys

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False):

        self.simulation_time = 0

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.carla.client.get_world().on_tick(self.on_world_tick)

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # 以1cm的间距取点
        #waypoints = self.carla.client.get_world().get_map().generate_waypoints(distance=1.0)
        
        # 道路标记
        #road_waypoints = []
        #for i in range(49, 50):
        #    for waypoint in waypoints:
        #        if waypoint.road_id == 49:
        #            road_waypoints.append(waypoint)
        #    self.draw_waypoints(self.carla.client.get_world(), road_waypoints, text=str(49), road_id=49)
        #    road_waypoints.clear()

    #def draw_waypoints(self, world, waypoints, text, road_id):
    #    for waypoint in waypoints:
    #        if waypoint.road_id == road_id:
    #            world.debug.draw_string(waypoint.transform.location, 
    #                                    text, 
    #                                    draw_shadow=False,
    #                                    color=CARLA.Color(r=0, g=255, b=0), 
    #                                    life_time=60.0)

    def on_world_tick(self, timestamp):
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self,number,last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,previous_generate,change_lane_dict,warning_dict,output_file):
        """
        Tick to simulation synchronization
        """

        # -----------------
        # sumo-->carla sync 将sumo的画面传入carla里，否则车辆只在sumo里出现, carla会不显示
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        hasNeighbors=False
        triggerCarlaWorldPlayerAuto=True
        constantVelocity=0
        if len(self.carla2sumo_ids)>0:
            carla_id = str(list(self.carla2sumo_ids.values())[0])
            
            print('carla_id:',carla_id)

            #current_time=datetime.timedelta(seconds=int(self.simulation_time))
            current_time=int(self.simulation_time)
            
            # 从第二个步长开始 且 未执行生成车辆代码
            if number>0:
                #生成交互车的名字，当前Carla车在sumo里的id
                newVehicleName='new_sumo_'+str(number)
                last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route_new,previous_flow,state,previous_generate=self.sumo.addVehicle(simulation_route,previous_flow,current_time,number,state,previous_generate,output_file,sumo_id=newVehicleName,carla_id=carla_id,last_edge_id=last_edge_id,midofEdgeGenerated=midofEdgeGenerated,endofEdgeGenerated=endofEdgeGenerated)
                simulation_route=simulation_route_new
                #add_check=self.sumo.addVehicle(sumo_id=newVehicleName,carla_id=carla_id)
                #if add_check:
                #    generated=True
            hasNeighbors,warning_dict,triggerCarlaWorldPlayerAuto,constantVelocity,remainingTime=self.sumo.findNeighbors(current_time,carla_id,number,0.05,warning_dict,output_file)

            #sun_altitude_angle: whether under autopilot model now True44 False45
            if remainingTime>=0 and self.carla.client.get_world().get_weather().sun_altitude_angle==44:
                self.carla.client.get_world().set_weather(CARLA.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=self.carla.client.get_world().get_weather().sun_azimuth_angle, 
                sun_altitude_angle=self.carla.client.get_world().get_weather().sun_altitude_angle, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=remainingTime, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.00000))
            elif remainingTime>=0 and self.carla.client.get_world().get_weather().sun_altitude_angle==45:
                self.carla.client.get_world().set_weather(CARLA.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=self.carla.client.get_world().get_weather().sun_azimuth_angle, 
                sun_altitude_angle=self.carla.client.get_world().get_weather().sun_altitude_angle, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=-1, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.00000))
                #self.carla.client.get_world().set_weather(CARLA.WeatherParameters(fog_distance=remainingTime))
            
        if hasNeighbors:
            change_lane_dict=self.sumo.ChangeLaneNeighbors(carla_id,current_time,number,output_file,self.carla.step_length,change_lane_dict)

        # Updating sumo actors in carla.
        #print('self.sumo2carla_ids:',self.sumo2carla_ids)
        for sumo_actor_id in self.sumo2carla_ids:
            #if sumo_actor_id=='sumo1111':
            #    print('continue')
            #    continue
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]
            #print('carla_actor_id:',carla_actor_id)

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            #constant_velocity_enabled=False
            #with open(r'C:\Users\Eddy\Desktop\constant_velocity_enabled.txt', 'a+') as f:
            #    #data = f.readline() if f.readline().strip()!="" else ""
            #    if f.readline()=='T':
            #        constant_velocity_enabled=True
                #print(data)

            #with open(r'C:\Users\Eddy\Desktop\constant_velocity_enabled.txt', 'w') as f:
            #    f.write(str(0))
            #WeatherParameters(cloudiness=10.000000, cloudiness=10.000000, precipitation=0.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=220.000000, sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000)
            #print(self.carla.client.get_world().get_weather().sun_azimuth_angle)

            #constant_velocity_enabled=False

            #if triggerCarlaWorldPlayerAuto==False:
            #sun_azimuth_angle: whether constant velocity mode; sun_altitude_angle: whether under autopilot model now
            if triggerCarlaWorldPlayerAuto==False and self.carla.client.get_world().get_weather().sun_azimuth_angle==221:
                if self.carla.client.get_world().get_weather().sun_altitude_angle==44:
                    #sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
                    #carla_spawned_actors = self.carla.spawned_actors
                    #print('numbers',len(set(self.carla2sumo_ids.values())))
                    #for sumo_actor_id in sumo_spawned_actors:
                    #print(set(self.carla2sumo_ids.values()))

                    print('AUTO Stopped....................')
                    #carla_actor = self.carla.get_actor(list(self.carla2sumo_ids.values())[0])
                    #carla_actor.set_autopilot(False)
                    #carla_actor.enable_constant_velocity(CARLA.Vector3D(constantVelocity, 0, 0))
                    #carla_actor.disable_constant_velocity()

            #print('carla_actor:',carla_actor)
            #print('sumo_actor:',sumo_actor)

            #print('carla_actor_id:',carla_actor_id)
            #print('sumo_actor_id:',sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals, carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None
            
            #self.sumo.findNeighbors(sumo_actor_id)
            
            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

            #self.sumo.findNeighbors(sumo_actor_id)
            #self.sumo.outputNeighbors(sumo_actor_id)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)
        return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,change_lane_dict,warning_dict,simulation_route,previous_flow,state,previous_generate]

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()

    #def core_function():
    #    hasNeighbors=False
            
    #    if len(synchronization.carla2sumo_ids)>0:
    #        carla_id = str(list(synchronization.carla2sumo_ids.values())[0])
            
    #        print('carla_id:',carla_id)
            
            # 从第二个步长开始 且 未执行生成车辆代码
    #        if number>0 and generated==False:
    #            #生成交互车的名字，当前Carla车在sumo里的id，生成交互车的初始速度25m/s
    #            add_check=synchronization.sumo.addVehicle('sumo1111',carla_id,30)
    #            if add_check:
    #                generated=True
    #        hasNeighbors=synchronization.sumo.findNeighbors(carla_id)
            
    #    if hasNeighbors:
    #        change_lane_dict=synchronization.sumo.ChangeLaneNeighbors(carla_id,number,args.step_length,change_lane_dict)

def synchronization_loop(args):
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)

    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)

    last_edge_id="-10086"
    midofEdgeGenerated=False
    endofEdgeGenerated=False
    
    number=0
    change_lane_dict={}
    warning_dict={}
    simulation_route=[46,40,16,7,-33,-24,-4,-1,10,40,16,7,-33,-24,-4,-1]
    previous_flow=[0,0,0]#previous_flow 0分流 1合流 2交织
    state=0 # 0手动 1监管 2娱乐
    previous_generate=0 #0交织区 1上 2下匝道口

    loca_name=time.strftime('%Y-%m-%d-%H-%M-%S')
    output_file=open(r'D:\Carla\Co-Simulation\Sumo\results\part1\sumo_'+str(loca_name)+'.txt',"w")

    try:
        while True:
            start = time.time()

            res=synchronization.tick(number,last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,previous_generate,change_lane_dict,warning_dict,output_file)

            last_edge_id=res[0]
            midofEdgeGenerated=res[1]
            endofEdgeGenerated=res[2]
            change_lane_dict=res[3]
            warning_dict=res[4]
            simulation_route=res[5]
            previous_flow=res[6]
            state=res[7]
            previous_generate=res[8]
            
            number=number+1

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')
        output_file.close()
        synchronization.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('sumo_cfg_file', type=str, help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.05,
                           #default=0.01,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synchronization_loop(arguments)
