# coding=gb2312
#!/usr/bin/env python
# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# Modify: Yichang Shao
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""
from __future__ import print_function

import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import winsound

import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- multiple imports -------------------------------------------------------------------
# ==============================================================================
import time
start = time.time()
import cv2


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
#from carla import Transform, Location, Rotation
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#实验室三联屏
VIEW_WIDTH = 1920//1
VIEW_HEIGHT = 1200//1

VIEW_HEIGHT_BEHIND = 480//1
VIEW_FOV = 90

#自己电脑
#VIEW_WIDTH = 960//1
#VIEW_HEIGHT = 540//1
#VIEW_FOV = 90

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def draw_waypoints(world, waypoints):
    for waypoint in waypoints:
        world.debug.draw_point(waypoint.transform.location, 
                                size=0.05, 
                                color=carla.Color(r=0, g=5, b=0), 
                                life_time=100)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        loca_name=time.strftime('%Y-%m-%d-%H-%M-%S')
        self.output_file=open(r'D:\Carla\Co-Simulation\Sumo\results\part1\carla_'+str(loca_name)+'.txt',"w")

        self.world_autopilot_changed=False
            
        self.hud = hud
        self.previous_remainingTime=-2
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        # multiple views
        self.right_camera = None
        self.left_camera = None
        self.behind_camera = None

        self.right_display = None
        self.left_display = None
        self.behind_display = None

        self.right_image = None
        self.left_image = None
        self.behind_image = None

        self.right_capture = True
        self.left_capture = True
        self.behind_capture = True

        self.right = None
        self.left = None
        self.behind = None

        self.current_road_id=-999
        #self.id_list=[46,128,40,163,16,93,7,78,33,48,24,178,4,133,1,138,35,53,29,143,11,193,17,63,32,113,15,203,44,173,23,98,43,118,22,103,42,188,45,168,36,68,26,83,39,198,38,123,8,148,31,108,25,158,3,73,5,183,12,58,0,88,2,153,46,128,40,163,16,93,7,78,30,68,26,83,39,198,38]
        self.id_list=[46,128,40,163,16,93,7,78,33,48,24,178,4,133,1,138,10,128,40,163,16,93,7,78,33,48,24,178,4,133,1,138,35]
        self.direction_txt='前方直行'

    def two_views_camera_blueprint(self):
	    three_views_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
	    three_views_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
	    three_views_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
	    three_views_camera_bp.set_attribute('fov', str(VIEW_FOV))
	    return three_views_camera_bp

    def behind_views_camera_blueprint(self):
	    behind_views_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
	    behind_views_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
	    behind_views_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT_BEHIND))
	    behind_views_camera_bp.set_attribute('fov', str(VIEW_FOV))
	    return behind_views_camera_bp

    def setup_three_views_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        
        right_camera_transform = carla.Transform(carla.Location(x=0.65, y=1, z=1.1), carla.Rotation(pitch=-3,yaw=150))
        self.right_camera = self.world.spawn_actor(self.two_views_camera_blueprint(), right_camera_transform, attach_to=self.player)
        weak_right_self = weakref.ref(self)
        self.right_camera.listen(lambda right_image: weak_right_self().set_right_image(weak_right_self, right_image))
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.right_camera.calibration = calibration
        
        left_camera_transform = carla.Transform(carla.Location(x=0.05, y=-1, z=1.1), carla.Rotation(pitch=-3,yaw=210))
        self.left_camera = self.world.spawn_actor(self.two_views_camera_blueprint(), left_camera_transform, attach_to=self.player)
        weak_left_self = weakref.ref(self)
        self.left_camera.listen(lambda left_image: weak_left_self().set_left_image(weak_left_self, left_image))
        self.left_camera.calibration = calibration
        
        behind_camera_transform = carla.Transform(carla.Location(x=-0.7, y=0, z=1.3), carla.Rotation(yaw=180))
        self.behind_camera = self.world.spawn_actor(self.behind_views_camera_blueprint(), behind_camera_transform, attach_to=self.player)
        weak_behind_self = weakref.ref(self)
        self.behind_camera.listen(lambda behind_image: weak_behind_self().set_behind_image(weak_behind_self, behind_image))
        calibration_b = np.identity(3)
        calibration_b[0, 2] = VIEW_WIDTH / 2.0
        calibration_b[1, 2] = VIEW_HEIGHT_BEHIND / 2.0
        calibration_b[0, 0] = calibration_b[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.behind_camera.calibration = calibration_b

    @staticmethod
    def set_right_image(weak_right_self, right_img):
	    """
	    Sets image coming from camera sensor.
	    The self.capture flag is a mean of synchronization - once the flag is
	    set, next coming image will be stored.
	    """
	    self = weak_right_self()
	    if self.right_capture:
	        self.right_image = right_img
	        self.right_capture = False

    @staticmethod
    def set_left_image(weak_left_self, left_img):
	    """
	    Sets image coming from camera sensor.
	    The self.capture flag is a mean of synchronization - once the flag is
	    set, next coming image will be stored.
	    """
	    self = weak_left_self()
	    if self.left_capture:
	        self.left_image = left_img
	        self.left_capture = False
    
    @staticmethod
    def set_behind_image(weak_behind_self, behind_img):
	    """
	    Sets image coming from camera sensor.
	    The self.capture flag is a mean of synchronization - once the flag is
	    set, next coming image will be stored.
	    """
	    self = weak_behind_self()
	    if self.behind_capture:
	        self.behind_image = behind_img
	        self.behind_capture = False

    def right_render(self, right_display):
        if self.right_image is not None:
            cv2.namedWindow("right_image",cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("right_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            i = np.array(self.right_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            self.right = i3
            self.right=cv2.flip(self.right,1)
            cv2.imshow("right_image", self.right)

    def left_render(self, left_display):
        if self.left_image is not None:
            cv2.namedWindow("left_image",cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("left_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            i = np.array(self.left_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            self.left = i3
            self.left=cv2.flip(self.left,1)
            cv2.imshow("left_image", self.left)

    def behind_render(self, behind_display):
        if self.behind_image is not None:
            cv2.namedWindow("behind_image",cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("behind_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            i = np.array(self.behind_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT_BEHIND, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            self.behind = i3
            self.behind=cv2.flip(self.behind,1)
            cv2.imshow("behind_image", self.behind)

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        #blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        # balck vehicle
        blueprint = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)[4]

        blueprint.set_attribute('role_name', self.actor_role_name)
        #print('role_name:', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            #color = random.choice(blueprint.get_attribute('color').recommended_values)
            color = blueprint.get_attribute('color').recommended_values[2]
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            #driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            driver_id = blueprint.get_attribute('driver_id').recommended_values[0]
            #print('driver_id:', driver_id)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            #spawn_point.location.z = 0.0
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            print('spawn_point number:',len(spawn_points))
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #spawn_point = spawn_points[1] if spawn_points else carla.Transform()
            #指定出生点
            #Transform(Location(x=-2005.250000, y=997.899963, z=4.000000), Rotation(pitch=0.000000, yaw=-89.999992, roll=0.000000))
            #yaw控制车头朝向
            spawn_point = carla.Transform(carla.Location(x=-2005.25, y=1150, z=4), carla.Rotation(pitch=0, yaw=-90, roll=0))
            print('spawn point:',spawn_point)
            
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot

        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE

            if self._autopilot_enabled:
                world.player.set_autopilot(self._autopilot_enabled)

                #world.hud.notification('Autopilot On')
                world.hud.notification('自动驾驶模式已开启')

                #enabled sun_altitude_angle true 44; false 45
                world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                sun_altitude_angle=44.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            client = carla.Client('127.0.0.1', 2000)
            traffic_manager = client.get_trafficmanager()

            ## 激进 可以超出限速100% 30->60
            traffic_manager.global_percentage_speed_difference(600)
            #traffic_manager.set_global_distance_to_leading_vehicle(50)
            #world.player.set_attribute('role_name', 'hero')
            #traffic_manager.set_hybrid_physics_mode(True)
            #traffic_manager.set_hybrid_physics_radius(50.0)

            #控制Carla车辆在自动驾驶情况下是否可以自由变道
            traffic_manager.auto_lane_change(world.player, True)
            #traffic_manager.random_left_lanechange_percentage(world.player,80)
            #traffic_manager.random_right_lanechange_percentage(world.player,20)
            traffic_manager.vehicle_percentage_speed_difference(world.player, -150.0)
            traffic_manager.distance_to_leading_vehicle(world.player, 5)

            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        try:
            # initialize steering wheel
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                raise ValueError("Please Connect Just One Joystick")

            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

            self._parser = ConfigParser()
            #For Thrustmaster T300
            self._parser.read('wheel_config.ini')
            self._steer_idx = int(
                self._parser.get('Thrustmaster T300', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('Thrustmaster T300', 'throttle'))
            self._brake_idx = int(self._parser.get('Thrustmaster T300', 'brake'))
            self._reverse_idx = int(self._parser.get('Thrustmaster T300', 'reverse'))
            self._handbrake_idx = int(
                self._parser.get('Thrustmaster T300', 'handbrake'))

            #For Logi G29
            #self._parser.read('wheel_config_logi.ini')
            #self._steer_idx = int(
            #    self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            #self._throttle_idx = int(
            #    self._parser.get('G29 Racing Wheel', 'throttle'))
            #self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            #self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            #self._handbrake_idx = int(
            #    self._parser.get('G29 Racing Wheel', 'handbrake'))
        except:
            print('No Joystick')

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights

        if world.world_autopilot_changed:
            self._autopilot_enabled = not self._autopilot_enabled

            world.player.set_autopilot(self._autopilot_enabled)

            #enabled sun_altitude_angle true 44; false 45
            if self._autopilot_enabled:
                #world.hud.notification('Autopilot On')
                world.hud.notification('自动驾驶模式已开启')
                world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                sun_altitude_angle=44.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            else:
                # notification will show in remianing time part
                #world.hud.notification('Autopilot Off')
                world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            world.world_autopilot_changed=False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                #if event.button == 0:
                #    world.restart()
                #elif event.button == 1:
                #    world.hud.toggle_info()
                if event.button == 3:
                    if not self._ackermann_enabled:
                        self._control.gear = 1 if self._control.reverse else -1
                    else:
                        self._ackermann_reverse *= -1
                        # Reset ackermann control
                        self._ackermann_control = carla.VehicleAckermannControl()
                elif event.button == 2:
                    #world.camera_manager.toggle_camera()
                    if not self._autopilot_enabled and not sync_mode:
                        print("WARNING: You are currently in asynchronous mode and could "
                              "experience some issues with the traffic simulation")
                    self._autopilot_enabled = not self._autopilot_enabled

                    world.player.set_autopilot(self._autopilot_enabled)

                    #enabled sun_altitude_angle true 44; false 45
                    if self._autopilot_enabled:
                        world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                        sun_altitude_angle=44.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                        rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
                    else:
                        #decelerate to 0 km/h after turning off autopilot mode
                        world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                        world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                        sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                        rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                    client = carla.Client('127.0.0.1', 2000)
                    traffic_manager = client.get_trafficmanager()
                    #traffic_manager.auto_lane_change(world.player, True)
                    traffic_manager.global_percentage_speed_difference(600)
                    traffic_manager.vehicle_percentage_speed_difference(world.player, -150.0)

                    #world.hud.notification(
                    #    'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

                    world.hud.notification(
                            '自动驾驶模式%s' % ('已开启' if self._autopilot_enabled else '已关闭'))
                #elif event.button == 3:
                #    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                #elif event.button == 23:
                #    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        self._autopilot_enabled = False
                        world.player.set_autopilot(False)

                        #decelerate to 0 km/h after turning off autopilot mode
                        world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                        #enabled sun_altitude_angle true 44; false 45
                        world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                        sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                        rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                        world.restart()

                        self._autopilot_enabled = True
                        world.player.set_autopilot(True)
                        #world.hud.notification('Autopilot On')
                        world.hud.notification('自动驾驶模式已开启')

                        #enabled sun_altitude_angle true 44; false 45
                        world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                        sun_altitude_angle=44.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                        rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                        client = carla.Client('127.0.0.1', 2000)
                        traffic_manager = client.get_trafficmanager()
                        #traffic_manager.auto_lane_change(world.player, True)
                        traffic_manager.global_percentage_speed_difference(600)
                        traffic_manager.vehicle_percentage_speed_difference(world.player, -150.0)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    #world.hud.notification('Autopilot Off')
                    world.hud.notification('自动驾驶模式已关闭')

                    #decelerate to 0 km/h after turning off autopilot mode
                    world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                    #enabled sun_altitude_angle true 44; false 45
                    world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                    sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                    rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled

                        world.player.set_autopilot(self._autopilot_enabled)

                        #enabled sun_altitude_angle true 44; false 45
                        if self._autopilot_enabled:
                            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                            sun_altitude_angle=44.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
                        else:
                            #decelerate to 0 km/h after turning off autopilot mode
                            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                        client = carla.Client('127.0.0.1', 2000)
                        traffic_manager = client.get_trafficmanager()
                        #traffic_manager.auto_lane_change(world.player, True)
                        traffic_manager.global_percentage_speed_difference(600)
                        traffic_manager.vehicle_percentage_speed_difference(world.player, -150.0)

                        #world.hud.notification(
                        #    'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

                        world.hud.notification(
                            '自动驾驶模式%s' % ('已开启' if self._autopilot_enabled else '已关闭'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.InteriorK_s
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                #self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time(), world)
                self._parse_vehicle_keys_autop(pygame.key.get_pressed(), clock.get_time(), world)
                try:
                    self._parse_vehicle_wheel_autop(world)
                except:
                    print('Warning: no wheel')
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))

                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time(), world)
                try:
                    self._parse_vehicle_wheel(world)
                except:
                    print('No Wheel')
                self._control.reverse = self._control.gear < 0
                 # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
                world.player.apply_control(self._control)

    #Specially for keys detection under autopilot mode
    def _parse_vehicle_keys_autop(self, keys, milliseconds, world):
        if keys[K_UP] or keys[K_w]:

            self._autopilot_enabled=False
            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:

            self._autopilot_enabled=False
            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:

            self._autopilot_enabled=False
            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:

            self._autopilot_enabled=False
            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)
    
    #Specially for keys detection under no-autopilot mode
    def _parse_vehicle_keys(self, keys, milliseconds, world):
        if keys[K_UP] or keys[K_w]:
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle, 
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle, 
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')            
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')            
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    #Specially for wheel detection under autopilot mode
    def _parse_vehicle_wheel_autop(self,world):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        v=world.player.get_velocity()

        #wheel left or right about 10 degree will trigger
        if abs(steerCmd)>=0.04 and (v.x+v.y)!=0:
            self._autopilot_enabled=False

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        if throttleCmd > 0 and (v.x+v.y)!=0:
            self._autopilot_enabled=False

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        if brakeCmd > 0 and (v.x+v.y)!=0:
            self._autopilot_enabled=False

            #enabled sun_altitude_angle true 44; false 45
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
            sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
            rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

            world.player.set_autopilot(False)
            #world.hud.notification('Autopilot Off')
            world.hud.notification('自动驾驶模式已关闭')

            #decelerate to 0 km/h after turning off autopilot mode
            world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    #Specially for wheel detection under no-autopilot mode
    def _parse_vehicle_wheel(self,world):
        
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        v=world.player.get_velocity()

        #wheel left or right 5 degree will trigger
        #if abs(steerCmd)>=0.025 and (v.x+v.y)!=0:
            #using wheather parameter 'sun_azimuth_angle' to pass parameters that control whether stop the constant velocity mode in run sync.py
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')            
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        #if throttleCmd > 0 and (v.x+v.y)!=0:
            #using wheather parameter 'sun_azimuth_angle' to pass parameters that control whether stop the constant velocity mode in run sync.py
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')            
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        #if brakeCmd > 0 and (v.x+v.y)!=0:
            #using wheather parameter 'sun_azimuth_angle' to pass parameters that control whether stop the constant velocity mode in run sync.py
            #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
            #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            #print('------------CANCEL Constant Velocity')            
            #world.player.disable_constant_velocity()
            #world.constant_velocity_enabled = False

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, waypoints):
        self.dim = (width, height)
        #font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font = pygame.font.Font('D:\Carla\PythonAPI\examples\ChineseBold.ttf',30)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

        #self._notifications = FadingText(font, (width/3, 60), (0, height - 60))
        self._notifications = FadingText(font, (400, 80), (width/2-200,40))
        self._remainingTimeNotifications = FadingText(font, (400, 80), (width/2-200,40))
        self._speedDisplay = FadingText(font, (200, 80), (width/2-80,height-100))
        self._directionDisplay = FadingText(font, (300, 80), (width/2+150,height-100))

        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

        self.waypoints=waypoints

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        self._remainingTimeNotifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -2.0, 2.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        nearby_vehicles=[]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles: (100m)']

            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 100.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                #self._info_text.append('% 4dm %s %s' % (d, vehicle_type, vehicle))
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                nearby_vehicles.append([vehicle.id,d])

        self._speedDisplay.set_green_text('% 3.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),seconds=0.5,font=pygame.font.Font(pygame.font.get_default_font(), 30))

        current_remainingTime=world.world.get_weather().wetness

        #Show remaining time only durning the autopilot mode
        if world.world.get_weather().sun_altitude_angle==44 and current_remainingTime!=-1:
            if current_remainingTime==world.previous_remainingTime:
                current_remainingTime=current_remainingTime-0.0333
            if current_remainingTime>3 and world.previous_remainingTime==-2:
            #if current_remainingTime>world.previous_remainingTime:
                winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_takeover.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)

            if 0.05<=current_remainingTime<=3 and world.previous_remainingTime!=-2:
            #if 0<current_remainingTime<=3 and world.previous_remainingTime!=-2:
                #self._remainingTimeNotifications.set_red_text('Remaining Time: %8.2f s' % current_remainingTime,seconds=0.5)
                self._remainingTimeNotifications.set_red_text('剩余接管时间: %8.2f 秒' % current_remainingTime,seconds=0.5)
                world.previous_remainingTime=current_remainingTime
            elif current_remainingTime>3:
                #self._remainingTimeNotifications.set_green_text('Remaining Time: %8.2f s' % current_remainingTime,seconds=0.5)
                self._remainingTimeNotifications.set_green_text('剩余接管时间: %8.2f 秒' % current_remainingTime,seconds=0.5)
                world.previous_remainingTime=current_remainingTime
            elif -0.05<current_remainingTime<0.05 and world.previous_remainingTime!=-2:
            #elif 0.05<=current_remainingTime<0.1 and world.previous_remainingTime==-2:
            #elif -0.05<=world.previous_remainingTime<=0:
                #self._remainingTimeNotifications.set_green_text('Switch to Autopolit and Stopped',seconds=2)
                self._remainingTimeNotifications.set_green_text('执行自动刹车指令',seconds=2)
                world.previous_remainingTime=-2

                world.world_autopilot_changed=True
                world.player.set_autopilot(False)

                #self._notifications.set_text('Autopilot Off',seconds=2)
                #self._notifications.set_text('自动驾驶模式已关闭',seconds=2)
                #world.hud.notification('Autopilot Off')

                #enabled sun_altitude_angle true 44; false 45
                world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=world.world.get_weather().sun_azimuth_angle, 
                sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=world.world.get_weather().wetness, scattering_intensity=1.000000, mie_scattering_scale=0.003996, 
                rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

                #decelerate to 0 km/h after turning off autopilot mode
                #world.player.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=1.0, hand_brake=False))

                #using wheather parameter 'sun_azimuth_angle' to pass parameters that control whether stop the constant velocity mode in run sync.py
                #world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=221.000000, sun_altitude_angle=world.world.get_weather().sun_altitude_angle,
                #fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
                #print('------------CANCEL Constant Velocity')
                #world.player.disable_constant_velocity()
                #world.constant_velocity_enabled = False

                #world.player.enable_constant_velocity(carla.Vector3D(0, 0, 0))
                #world.constant_velocity_enabled = True
            #else:
            #    world.previous_remainingTime=-2
        else:
            #current_remainingTime=-1
            world.previous_remainingTime=-2

        id_list=world.id_list
        road_waypoints = []

        junction_id_list=[78,48,68,168,83,188,198,103,123,118,98,148,173,108,203,158,113,73,63,183,193,58,143,88,53,153,138,128,133,163,93,178]
        
        outer_id_list=[40,16,7,30,26,39,38,8,31,25,3,5,12,0,2,46]
        inner_id_list=[1,4,24,41,45,42,22,43,23,44,15,32,17,11,29,35]

        normal_id_list=outer_id_list+inner_id_list
        ramp_id_list=[33,36,34,19,14,9,13,28,20,37,27,18,47,10,6,21]

        current_waypoint = world.world.get_map().get_waypoint(t.location)

        if current_waypoint.road_id!=None:
            road_id = current_waypoint.road_id
            if road_id!=world.current_road_id:
                world.current_road_id=road_id
                index=-1
                try:
                    index=id_list.index(road_id)
                except:
                    index=-1

                if 0<=index<len(id_list)-2:
                    if id_list[index+1] in junction_id_list:
                        if id_list[index+2] in ramp_id_list:
                            winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_ramp.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                            world.direction_txt='前方驶入匝道'
                        else:
                            winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_keep.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                            world.direction_txt='前方直行'
                        for waypoint in self.waypoints:
                            if waypoint.junction_id == id_list[index+1] and id_list[index+2] in ramp_id_list and waypoint.lane_id in [1,-1]:
                                    road_waypoints.append(waypoint)
                            elif waypoint.junction_id == id_list[index+1] and id_list[index+2] in normal_id_list:
                                if id_list[index] not in ramp_id_list and waypoint.lane_id in [2,3,4,-2,-3,-4]:
                                    road_waypoints.append(waypoint)
                                elif id_list[index] in ramp_id_list and waypoint.lane_id in [1,-1]:
                                    road_waypoints.append(waypoint)
                            
                elif index>=len(id_list)-2:
                    print('finished')
                    self._notifications.set_text('实验结束',seconds=5)
                    winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\end_exp.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                if index!=-1:
                    del world.id_list[0:index]
                print(world.id_list)


        draw_waypoints(world.world, road_waypoints)
        road_waypoints.clear()

        self._directionDisplay.set_green_text(world.direction_txt,seconds=0.5)


        # print record info of the carla vehicle here
        print(str(self.frame)+'#'+str(int(self.simulation_time))+'#'+str((v.x, v.y, v.z))+'#'+str(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))+'#'+str((compass, heading))+'#'
                +str((t.location.x, t.location.y))+'#'+str(c.steer)+'#'+str(c.throttle)+'#'+str(c.brake)+'#'+str(c.reverse)+'#'+str(collision[-1])+'#'+str(nearby_vehicles)+'#'+str((current_remainingTime,world.previous_remainingTime)),file=world.output_file)

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self._remainingTimeNotifications.render(display)
        self._speedDisplay.render(display)
        self._directionDisplay.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        text_rect = text_texture.get_rect(center = self.surface.get_rect().center)
        self.surface.blit(text_texture, text_rect)

    def set_green_text(self, text, color=(0, 255, 0), seconds=2.0,font=0):
        text_texture = self.font.render(text, True, color)
        if font!=0:
            text_texture = font.render(text, True, color)
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        text_rect = text_texture.get_rect(center = self.surface.get_rect().center)
        self.surface.blit(text_texture, text_rect)

    def set_red_text(self, text, color=(255, 0, 0), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        text_rect = text_texture.get_rect(center = self.surface.get_rect().center)
        self.surface.blit(text_texture, text_rect)

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        #self.hud.notification('Collision with %r' % actor_type)
        self.hud.notification('发生碰撞: %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        
        
        #Marks
        
        #self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        #self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=0)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType
        # x越小越靠后排,+y越大越靠右,-y越大越靠左
        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                #实验室三联屏
                (carla.Transform(carla.Location(x=+0.009*bound_x, y=-0.12*bound_y, z=0.99*bound_z),carla.Rotation(pitch=-3)),Attachment.Rigid),
                (carla.Transform(carla.Location(x=+0.1*bound_x, y=-0.12*bound_y, z=0.99*bound_z),carla.Rotation(pitch=-3)),Attachment.Rigid),
                (carla.Transform(carla.Location(x=+0.042*bound_x, y=-0.1*bound_y, z=0.99*bound_z)), Attachment.Rigid),
                #自己电脑
                #(carla.Transform(carla.Location(x=+0.03*bound_x, y=-0.25*bound_y, z=1.0*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+0.25*bound_x, y=0*bound_y, z=1.0*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=0, z=3), carla.Rotation(pitch=-20,yaw=180,roll=0)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))

                #实验室三联屏
                #bp.set_attribute('fov', str(150))
                bp.set_attribute('fov', str(133))

                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        #client.start_recorder(r'D:\Carla\Co-Simulation\Sumo\exp1.log')

        sim_world = client.get_world()
        #sim_world.unload_map_layer(carla.MapLayer.All)

        # 以10cm的间距取点
        waypoints = sim_world.get_map().generate_waypoints(distance=10.0)

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height, waypoints)
        world = World(sim_world, hud, args)

        # right, left, behind views
        world.setup_three_views_camera()
        #world.right_display = cv2.namedWindow('right')
        #world.left_display = cv2.namedWindow('left')
        #world.behind_display = cv2.namedWindow('behind')

        world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=220.000000, sun_altitude_angle=45.000000, 
        fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))

        #controller = KeyboardControl(world, args.autopilot)
        controller = DualControl(world, args.autopilot)

        #choices=["cautious", "normal", "aggressive"]
        #agent = BehaviorAgent(world.player, behavior='aggressive')
        #spawn_points = world.map.get_spawn_points()
        #destination = random.choice(spawn_points).location
        #agent.set_destination(destination)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
            #if controller.parse_events(world, clock):
                return
            
            #control = agent.run_step()
            #world.player.apply_control(control)
            
            world.tick(clock)

            world.right_capture = True
            world.left_capture = True
            world.behind_capture = True

            world.render(display)
            pygame.display.flip()

            #pygame.event.pump()

            world.right_render(world.right_display)
            world.left_render(world.left_display)
            world.behind_render(world.behind_display)

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.player.get_world().set_weather(carla.WeatherParameters(cloudiness=10.000000, precipitation_deposits=0.000000, wind_intensity=5.000000, sun_azimuth_angle=220.000000, sun_altitude_angle=45.000000, 
            fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.250000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.003996, rayleigh_scattering_scale=0.033100, dust_storm=0.000000))
            
            world.right_camera.destroy()
            world.left_camera.destroy()
            world.behind_camera.destroy()

            world.output_file.close()
            
            world.destroy()

        pygame.quit()

        cv2.destroyAllWindows()


def constant_velocity_enabled():
    return constant_velocity_enabled

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        #default=True,
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        #实验室三联屏
        default='4096x720',
        #自己电脑
        #default='1728x972',
        #default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        #default=True,
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
