# -*- coding: utf-8 -*-
#!/usr/bin/env python
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# Modify: Yichang Shao
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the sumo simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import winsound
import time
import collections
import enum
import logging
import os
import math
import random
import carla  # pylint: disable=import-error
import sumolib  # pylint: disable=import-error
import traci  # pylint: disable=import-error

from .constants import INVALID_ACTOR_ID

import lxml.etree as ET  # pylint: disable=import-error

# ==================================================================================================
# -- sumo definitions ------------------------------------------------------------------------------
# ==================================================================================================


# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
class SumoSignalState(object):
    """
    SumoSignalState contains the different traffic light states.
    """
    RED = 'r'
    YELLOW = 'y'
    GREEN = 'G'
    GREEN_WITHOUT_PRIORITY = 'g'
    GREEN_RIGHT_TURN = 's'
    RED_YELLOW = 'u'
    OFF_BLINKING = 'o'
    OFF = 'O'


# https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html
class SumoVehSignal(object):
    """
    SumoVehSignal contains the different sumo vehicle signals.
    """
    BLINKER_RIGHT = 1 << 0
    BLINKER_LEFT = 1 << 1
    BLINKER_EMERGENCY = 1 << 2
    BRAKELIGHT = 1 << 3
    FRONTLIGHT = 1 << 4
    FOGLIGHT = 1 << 5
    HIGHBEAM = 1 << 6
    BACKDRIVE = 1 << 7
    WIPER = 1 << 8
    DOOR_OPEN_LEFT = 1 << 9
    DOOR_OPEN_RIGHT = 1 << 10
    EMERGENCY_BLUE = 1 << 11
    EMERGENCY_RED = 1 << 12
    EMERGENCY_YELLOW = 1 << 13


# https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class
class SumoActorClass(enum.Enum):
    """
    SumoActorClass enumerates the different sumo actor classes.
    """
    IGNORING = "ignoring"
    PRIVATE = "private"
    EMERGENCY = "emergency"
    AUTHORITY = "authority"
    ARMY = "army"
    VIP = "vip"
    PEDESTRIAN = "pedestrian"
    PASSENGER = "passenger"
    HOV = "hov"
    TAXI = "taxi"
    BUS = "bus"
    COACH = "coach"
    DELIVERY = "delivery"
    TRUCK = "truck"
    TRAILER = "trailer"
    MOTORCYCLE = "motorcycle"
    MOPED = "moped"
    BICYCLE = "bicycle"
    EVEHICLE = "evehicle"
    TRAM = "tram"
    RAIL_URBAN = "rail_urban"
    RAIL = "rail"
    RAIL_ELECTRIC = "rail_electric"
    RAIL_FAST = "rail_fast"
    SHIP = "ship"
    CUSTOM1 = "custom1"
    CUSTOM2 = "custom2"


SumoActor = collections.namedtuple('SumoActor', 'type_id vclass transform signals extent color')

# ==================================================================================================
# -- sumo traffic lights ---------------------------------------------------------------------------
# ==================================================================================================


class SumoTLLogic(object):
    """
    SumoTLLogic holds the data relative to a traffic light in sumo.
    """
    def __init__(self, tlid, states, parameters):
        self.tlid = tlid
        self.states = states

        self._landmark2link = {}
        self._link2landmark = {}
        for link_index, landmark_id in parameters.items():
            # Link index information is added in the parameter as 'linkSignalID:x'
            link_index = int(link_index.split(':')[1])

            if landmark_id not in self._landmark2link:
                self._landmark2link[landmark_id] = []
            self._landmark2link[landmark_id].append((tlid, link_index))
            self._link2landmark[(tlid, link_index)] = landmark_id

    def get_number_signals(self):
        """
        Returns number of internal signals of the traffic light.
        """
        if len(self.states) > 0:
            return len(self.states[0])
        return 0

    def get_all_signals(self):
        """
        Returns all the signals of the traffic light.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return [(self.tlid, i) for i in range(self.get_number_signals())]

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with this traffic light.
        """
        return self._landmark2link.keys()

    def get_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return self._landmark2link.get(landmark_id, [])


class SumoTLManager(object):
    """
    SumoTLManager is responsible for the management of the sumo traffic lights (i.e., keeps control
    of the current program, phase, ...)
    """
    def __init__(self):
        self._tls = {}  # {tlid: {program_id: SumoTLLogic}
        self._current_program = {}  # {tlid: program_id}
        self._current_phase = {}  # {tlid: index_phase}

        for tlid in traci.trafficlight.getIDList():
            self.subscribe(tlid)

            self._tls[tlid] = {}
            for tllogic in traci.trafficlight.getAllProgramLogics(tlid):
                states = [phase.state for phase in tllogic.getPhases()]
                parameters = tllogic.getParameters()
                tl = SumoTLLogic(tlid, states, parameters)
                self._tls[tlid][tllogic.programID] = tl

            # Get current status of the traffic lights.
            self._current_program[tlid] = traci.trafficlight.getProgram(tlid)
            self._current_phase[tlid] = traci.trafficlight.getPhase(tlid)

        self._off = False

    @staticmethod
    def subscribe(tlid):
        """
        Subscribe the given traffic ligth to the following variables:

            * Current program.
            * Current phase.
        """
        traci.trafficlight.subscribe(tlid, [
            traci.constants.TL_CURRENT_PROGRAM,
            traci.constants.TL_CURRENT_PHASE,
        ])

    @staticmethod
    def unsubscribe(tlid):
        """
        Unsubscribe the given traffic ligth from receiving updated information each step.
        """
        traci.trafficlight.unsubscribe(tlid)

    def get_all_signals(self):
        """
        Returns all the traffic light signals.
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_all_signals())
        return signals

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with a traffic light in the simulation.
        """
        landmarks = set()
        for tlid, program_id in self._current_program.items():
            landmarks.update(self._tls[tlid][program_id].get_all_landmarks())
        return landmarks

    def get_all_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_associated_signals(landmark_id))
        return signals

    def get_state(self, landmark_id):
        """
        Returns the traffic light state of the signals associated with the given landmark.
        """
        states = set()
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            current_program = self._current_program[tlid]
            current_phase = self._current_phase[tlid]

            tl = self._tls[tlid][current_program]
            states.update(tl.states[current_phase][link_index])

        if len(states) == 1:
            return states.pop()
        elif len(states) > 1:
            logging.warning('Landmark %s is associated with signals with different states',
                            landmark_id)
            return SumoSignalState.RED
        else:
            return None

    def set_state(self, landmark_id, state):
        """
        Updates the state of all the signals associated with the given landmark.
        """
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            traci.trafficlight.setLinkState(tlid, link_index, state)
        return True

    def switch_off(self):
        """
        Switch off all traffic lights.
        """
        for tlid, link_index in self.get_all_signals():
            traci.trafficlight.setLinkState(tlid, link_index, SumoSignalState.OFF)
        self._off = True

    def tick(self):
        """
        Tick to traffic light manager
        """
        if self._off is False:
            for tl_id in traci.trafficlight.getIDList():
                results = traci.trafficlight.getSubscriptionResults(tl_id)
                current_program = results[traci.constants.TL_CURRENT_PROGRAM]
                current_phase = results[traci.constants.TL_CURRENT_PHASE]

                if current_program != 'online':
                    self._current_program[tl_id] = current_program
                    self._current_phase[tl_id] = current_phase


# ==================================================================================================
# -- sumo simulation -------------------------------------------------------------------------------
# ==================================================================================================

def _get_sumo_net(cfg_file):
    """
    Returns sumo net.

    This method reads the sumo configuration file and retrieve the sumo net filename to create the
    net.
    """
    cfg_file = os.path.join(os.getcwd(), cfg_file)

    tree = ET.parse(cfg_file)
    tag = tree.find('//net-file')
    if tag is None:
        return None

    net_file = os.path.join(os.path.dirname(cfg_file), tag.get('value'))
    logging.debug('Reading net file: %s', net_file)

    sumo_net = sumolib.net.readNet(net_file)
    return sumo_net

class SumoSimulation(object):
    """
    SumoSimulation is responsible for the management of the sumo simulation.
    """
    def __init__(self, cfg_file, step_length, host=None, port=None, sumo_gui=False, client_order=1):
        if sumo_gui is True:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        if host is None or port is None:
            logging.info('Starting new sumo server...')
            if sumo_gui is True:
                logging.info('Remember to press the play button to start the simulation')

            traci.start([sumo_binary,
                '--configuration-file', cfg_file,
                '--step-length', str(step_length),
                '--lateral-resolution', '0.25',
                '--collision.check-junctions'
            ])

        else:
            logging.info('Connection to sumo server. Host: %s Port: %s', host, port)
            traci.init(host=host, port=port)

        traci.setOrder(client_order)

        # Retrieving net from configuration file.
        self.net = _get_sumo_net(cfg_file)

        # To keep track of the vehicle classes for which a route has been generated in sumo.
        self._routes = set()

        # Variable to asign an id to new added actors.
        self._sequential_id = 0

        # Structures to keep track of the spawned and destroyed vehicles at each time step.
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Traffic light manager.
        self.traffic_light_manager = SumoTLManager()


        #timedown
        self.timedown=10

    @property
    def traffic_light_ids(self):
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id):
        """
        Subscribe the given actor to the following variables:

            * Type.
            * Vehicle class.
            * Color.
            * Length, Width, Height.
            * Position3D (i.e., x, y, z).
            * Angle, Slope.
            * Speed.
            * Lateral speed.
            * Signals.
        """
        traci.vehicle.subscribe(actor_id, [
            traci.constants.VAR_TYPE, traci.constants.VAR_VEHICLECLASS, traci.constants.VAR_COLOR,
            traci.constants.VAR_LENGTH, traci.constants.VAR_WIDTH, traci.constants.VAR_HEIGHT,
            traci.constants.VAR_POSITION3D, traci.constants.VAR_ANGLE, traci.constants.VAR_SLOPE,
            traci.constants.VAR_SPEED, traci.constants.VAR_SPEED_LAT, traci.constants.VAR_SIGNALS
        ])

    @staticmethod
    def unsubscribe(actor_id):
        """
        Unsubscribe the given actor from receiving updated information each step.
        """
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self):
        """
        Accessor for sumo net offset.
        """
        if self.net is None:
            return (0, 0)
        return self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id):
        """
        Accessor for sumo actor.
        """
        results = traci.vehicle.getSubscriptionResults(actor_id)

        type_id = results[traci.constants.VAR_TYPE]
        vclass = SumoActorClass(results[traci.constants.VAR_VEHICLECLASS])
        color = results[traci.constants.VAR_COLOR]

        length = results[traci.constants.VAR_LENGTH]
        width = results[traci.constants.VAR_WIDTH]
        height = results[traci.constants.VAR_HEIGHT]

        location = list(results[traci.constants.VAR_POSITION3D])
        rotation = [results[traci.constants.VAR_SLOPE], results[traci.constants.VAR_ANGLE], 0.0]
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]),
                                    carla.Rotation(rotation[0], rotation[1], rotation[2]))

        signals = results[traci.constants.VAR_SIGNALS]
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        return SumoActor(type_id, vclass, transform, signals, extent, color)

    def spawn_actor(self, type_id, color=None):
        """
        Spawns a new actor.

            :param type_id: vtype to be spawned.
            :param color: color attribute for this specific actor.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        actor_id = 'carla' + str(self._sequential_id)
        try:
            vclass = traci.vehicletype.getVehicleClass(type_id)
            if vclass not in self._routes:
                logging.debug('Creating route for %s vehicle class', vclass)
                allowed_edges = [e for e in self.net.getEdges() if e.allows(vclass)]
                if allowed_edges:
                    traci.route.add("carla_route_{}".format(vclass), [allowed_edges[0].getID()])
                    self._routes.add(vclass)
                else:
                    logging.error(
                        'Could not found a route for %s. No vehicle will be spawned in sumo',
                        type_id)
                    return INVALID_ACTOR_ID

            traci.vehicle.add(actor_id, 'carla_route_{}'.format(vclass), typeID=type_id)
        except traci.exceptions.TraCIException as error:
            logging.error('Spawn sumo actor failed: %s', error)
            return INVALID_ACTOR_ID

        if color is not None:
            color = color.split(',')
            traci.vehicle.setColor(actor_id, color)

        self._sequential_id += 1

        return actor_id

    @staticmethod
    def destroy_actor(actor_id):
        """
        Destroys the given actor.
        """
        traci.vehicle.remove(actor_id)

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        self.traffic_light_manager.switch_off()

    #核心：寻找周围车辆并判断是否执行换道操作
    def findNeighbors(self, current_time, vehicle_id, number, step_length, warning_dict,output_file):
        # 监听前后30m范围内 左右当前三车道内的车辆信息
        traci.vehicle.subscribeContext(vehicle_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 0, [
                traci.constants.VAR_SPEED,# speed m/s
                traci.constants.VAR_ACCELERATION,#acceleration, also deceleration, m2/s？ need to be checked again
                #traci.constants.VAR_EDGES,#for example('49',)
                traci.constants.VAR_LANE_ID,#str, edge id and lane id, for example '49_2'
                traci.constants.VAR_LANEPOSITION,#the distance between current location and start location of this lane(edge)
                #traci.constants.VAR_LANE_INDEX,#int, land id, for example 2
                traci.constants.VAR_POSITION,#x,y location
                traci.constants.VAR_LENGTH,#vehicle's length
                traci.constants.VAR_MINGAP])#vehicle's mingap
        traci.vehicle.addSubscriptionFilterLanes([-2,-1,0,1,2], noOpposite=True, downstreamDist=150, upstreamDist=150)

        triggerCarlaWorldPlayerAuto=True
        constantVelocity=0
        #输出监听内容
        content=traci.vehicle.getContextSubscriptionResults(vehicle_id)


        # !!!!在这里记录数据
        #with open(r'C:\Users\Eddy\Desktop\output_exp1.txt', 'a+') as f:
        #    f.write(str(number)+'#'+str(round(number*step_length,2))+'#'+str(content))
        #    f.writelines("\n")
        print(str(current_time)+'#'+str(number)+'#'+str(round(number*step_length,2))+'#'+str(time.time())+'#'+str(content),file=output_file)

        remainingTime=-1

        #大于1是除去Carla本车
        if len(content)>1:
            #判断30m内是否有前车
            leader=traci.vehicle.getLeader(vehicle_id,dist=30)

            if leader!=None:
                #Carla本车行驶信息
                carla_speed=content[vehicle_id][64]
                carla_pos=content[vehicle_id][66]
                carla_len=content[vehicle_id][68]
                #Carla本车所在道路信息
                #lane_id=traci.vehicle.getLaneIndex(vehicle_id)
                #road_id=traci.vehicle.getRoadID(vehicle_id)
                #获取字典里每一辆周围车
                for k,v in content.items():
                    #找交互影响车辆
                    #这里的同一edge有问题，因为有交接道路不同edge
                    #判断不是carla本车，且必须是同一edge，同一车道
                    #if k != vehicle_id and traci.vehicle.getRoadID(k)==road_id and traci.vehicle.getLaneIndex(k)==lane_id:
                    
                    #直接判断车ID是否为getLeader()结果
                    if k==str(leader[0]):
                        print('-------------',leader[0])
                        #获取sumo车的速度，长度，二维位置坐标
                        sumo_speed=v[64]
                        sumo_len=v[68]
                        sumo_pos=v[66]

                        #待第2次确认：carla车辆点位在车头中间
                        #计算车辆距离
                        distance=math.sqrt((carla_pos[0]-sumo_pos[0])**2+(carla_pos[1]-sumo_pos[1])**2)-sumo_len
                        #计算TTC
                        ttc=-999
                        if carla_speed!=sumo_speed:
                            ttc=distance/(carla_speed-sumo_speed)
                        #print(k,'distance:',distance,'long/lat distance:',carla_pos[0]-sumo_pos[0],carla_pos[1]-sumo_pos[1],'vehicle:',carla_len,sumo_len) 
                        
                        print('SUMO_ID: ',k,'ttc:',ttc,'distance m:',distance,'carla speed m/s:',carla_speed,'sumo speed m/s:', sumo_speed)

                        if 0<ttc<=2.7:
                            remainingTime=0
                            print('warning time: 0')

                        elif 2.7<ttc<=10:
                            #if k not in warning_dict or (number*step_length-warning_dict[k])>0:
                            if k not in warning_dict:
                                print('warning time remining: ',ttc-2.7)
                                warning_dict[k]=number*step_length+ttc-2.7
                                
                                remainingTime=ttc-2.7

                                triggerCarlaWorldPlayerAuto=False
                                if sumo_speed>5:
                                    constantVelocity=sumo_speed+1
                                else:
                                    constantVelocity=0 
                                    #self.timedown=number*step_length+ttc-2.7

                            elif k in warning_dict and (number*step_length-warning_dict[k])<0:
                                print('warning time remaining: ',warning_dict[k]-number*step_length)

                                remainingTime=warning_dict[k]-number*step_length

                                triggerCarlaWorldPlayerAuto=False
                                if sumo_speed>5:
                                    constantVelocity=sumo_speed+1
                                else:
                                    constantVelocity=0
                                    #self.timedown=number*step_length+ttc-2.7
                            #elif k in warning_dict and (number*step_length-warning_dict[k])>=0:
                            #    warning_dict.pop(k)
                            #triggerCarlaWorldPlayerAuto=False
                            #if sumo_speed>5:
                            #    constantVelocity=sumo_speed+1
                            #else:
                            #    constantVelocity=0
                            #traci.vehicle.changeLane(vehicle_id,1,10)
                            #traci.vehicle.slowDown(vehicle_id,0,step_length)
                            #set_autopilot(triggerCarlaWorldPlayerAuto)
                        else:
                            if k in warning_dict and (number*step_length-warning_dict[k])<0:
                                print('warning time remaining: ',warning_dict[k]-number*step_length)

                                remainingTime=warning_dict[k]-number*step_length

                                triggerCarlaWorldPlayerAuto=False
                                if sumo_speed>5:
                                    constantVelocity=sumo_speed+1
                                else:
                                    constantVelocity=0
                            #elif k in warning_dict and (number*step_length-warning_dict[k])>=0:
                            #    warning_dict.pop(k)
                            
                        if remainingTime!=-1:
                            #print('SUMO_ID: ',k,'ttc:',ttc,'distance m:',distance,'carla speed m/s:',carla_speed,'sumo speed m/s:', sumo_speed)
                            print('#'+str(current_time)+'#'+str(number)+'#'+str(round(number*step_length,2))+'#'+str(k)+'#'+str(distance)+'#'+str(carla_speed)+'#'+str(sumo_speed)+'#'+str(ttc)+'#'+str(remainingTime),file=output_file)
                            
            return True,warning_dict,triggerCarlaWorldPlayerAuto,constantVelocity,remainingTime
        return False,warning_dict,triggerCarlaWorldPlayerAuto,constantVelocity,remainingTime
        #print('leader',traci.vehicle.getLeader(vehicle_id))
        #print('leftleader',traci.vehicle.getLeftLeaders(vehicle_id, False))
        #print('rightleader',traci.vehicle.getRightLeaders(vehicle_id))
        #print('leftfollower',traci.vehicle.getLeftFollowers(vehicle_id))
        #print('rightfollower',traci.vehicle.getRightFollowers(vehicle_id))

    #核心：执行换道决策
    def ChangeLaneNeighbors(self,vehicle_id,current_time,number,output_file,step_length,change_lane_dict):
        #监听车辆
        content=traci.vehicle.getContextSubscriptionResults(vehicle_id)

        #大于1是除去Carla本车
        #if len(content)>1:
            #打印周围车辆信息
        #print(content)
        
        lane_id=traci.vehicle.getLaneIndex(vehicle_id)
        road_id=traci.vehicle.getRoadID(vehicle_id)
        #print('carla_id: ',vehicle_id,'lane_id: ',lane_id,'road_id: ',road_id)

        #获取Carla的车辆长度，mingap，二维位置坐标
        carla_len=content[vehicle_id][68]
        carla_mingap=content[vehicle_id][76]
        carla_pos=content[vehicle_id][66]
        carla_speed=content[vehicle_id][64]

        carla_lane_pos=traci.vehicle.getLanePosition(vehicle_id)

        #next_edge=traci.vehicle.getNextLinks(vehicle_id)
        #print('next_edge:',next_edge)
        
        leader=traci.vehicle.getLeader(vehicle_id,300)
        leftLeader=traci.vehicle.getLeftLeaders(vehicle_id)
        rightLeader=traci.vehicle.getRightLeaders(vehicle_id)
        print('leftLeader: ',leftLeader)
        print('rightLeader: ',rightLeader)
        print('Leader: ',leader,type(leader))

        leftRightLeaderList=[]
        if leftLeader!=None and len(leftLeader)!=0:
            leftRightLeaderList.append([leftLeader[0][0],leftLeader[0][1]])
        if rightLeader!=None and len(rightLeader)!=0:
            leftRightLeaderList.append([rightLeader[0][0],rightLeader[0][1]])

        #获取字典里每一辆周围车
        for k,v in content.items():
            if k.endswith(('+1','+2','+3','+4')):
                continue
            #这里的同一edge有问题，因为有交接道路不同edge
            #判断不是carla本车，必须是同一edge，不在同一车道
            for j in leftRightLeaderList:
            #if k in leftRightLeaderList:
            #if k != vehicle_id and traci.vehicle.getRoadID(k)==road_id and traci.vehicle.getLaneIndex(k)!=lane_id:
                sumo_lane_pos=traci.vehicle.getLanePosition(k)
                if (sumo_lane_pos-carla_lane_pos)>50:
                    continue

                if leader != None and k==leader[0] and k in change_lane_dict and change_lane_dict[k][1]==-1:
                    slow_speed=random.uniform(2,4)
                    slow_duration=random.uniform(4,5.5)
                    traci.vehicle.slowDown(k,slow_speed,slow_duration)
                    change_lane_dict[k][1]==0
                    print('S'+str(current_time)+'#'+str(number)+'#'+str(round(number*step_length,2))+'#'+str(k)+'#'+str(slow_speed)+'#'+str(slow_duration),file=output_file)
                if k==j[0]:
                    if k in change_lane_dict and (number*step_length-change_lane_dict[k][0])<5:
                        print('this vehicle lane change request has been runned')
                        print('Current Time: ',number*step_length,'Time Interval: ',number*step_length-change_lane_dict[k][0])
                    else:
                        #获取周围sumo车的车辆长度，mingap，二维位置坐标
                        sumo_len=v[68]
                        sumo_speed=v[64]
                        #sumo_mingap=v[76]
                        #sumo_pos=v[66]

                        ttc=-999
                        if carla_speed!=sumo_speed:
                            ttc=j[1]/(carla_speed-sumo_speed)

                        #if ttc>0:
                        #    print('预警倒计时：',math.ceil(ttc)+5)


                        #Carla车与sumo车初始距离
                        #distance=0

                        #当Carla车在sumo车前方的情况下
                        #if carla_pos[0]>sumo_pos[0]:
                            #两车间距=两车间距的绝对值-carla车长+sumo车的mingap
                            #distance=abs(carla_pos[0]-sumo_pos[0])-carla_len+sumo_mingap
                            #print(k,'distance:',distance,'long/lat distance:',carla_pos[0]-sumo_pos[0],carla_pos[1]-sumo_pos[1],'vehicle:',carla_len,sumo_len)
                        #当Carla车在sumo车后方的情况下
                        #else:
                            #两车间距=两车间距的绝对值-sumo车长+carla车的mingap
                            #distance=abs(carla_pos[0]-sumo_pos[0])-sumo_len+carla_mingap
                            #print(k,'distance:',distance,'long/lat distance:',carla_pos[0]-sumo_pos[0],carla_pos[1]-sumo_pos[1],'vehicle:',carla_len,sumo_len)

                            #lane_change_state=traci.vehicle.getLaneChangeState(k,-1)
                            #if lane_change_state!=(1073741824,1073741824):
                            #print('lane_change_state:',lane_change_state)
                            #last_action_time=traci.vehicle.getLastActionTime(k)
                            #departure_time=traci.vehicle.getDeparture(k)
                            #print('last_action_time:',last_action_time)
                            #print('departure_time:',departure_time)
                            #print('current_time:',number*0.05)
                            #print('current_time_carla:',current_time_carla,'current_time_sumo:',current_time_sumo)

                        #sumo车设置成不尊重后车模式
                        traci.vehicle.setLaneChangeMode(k,0b010101010001)
                        #执行换道操作
                        changing_duration=random.uniform(2,3.5)
                        traci.vehicle.changeLane(k,lane_id,changing_duration)

                        #traci.vehicle.setStop(vehID=k,edgeID=road_id,laneIndex=lane_id,pos=traci.vehicle.getLanePosition(vehicle_id)+150,duration=20)

                        print('C'+str(current_time)+'#'+str(number)+'#'+str(round(number*step_length,2))+'#'+str(k)+'#'+str(lane_id)+'#'+str(changing_duration),file=output_file)

                        #traci.vehicle.slowDown(k,5,10)

                        print('lane change runned')
                        change_lane_dict[k]=[number*step_length,-1]
        return change_lane_dict               

    #核心：主动式生成新交互车
    def addVehicle(self,simulation_route,previous_flow,current_time,number,state,previous_generate,output_file,sumo_id,carla_id,last_edge_id,midofEdgeGenerated,endofEdgeGenerated):
        #print(carla_id,": getRoute: ",traci.vehicle.getRoute(carla_id))
        #print(carla_id,": getDistance: ",traci.vehicle.getDistance(carla_id))
        #print(carla_id,": getPosition: ",traci.vehicle.getPosition(carla_id))
        #print(carla_id,": getDrivingDistance2D: ",traci.vehicle.getDrivingDistance2D(carla_id,-2000.0,-1000.0))
        #print(carla_id,": getDrivingDistance: ",traci.vehicle.getDrivingDistance(carla_id,"27",1,laneIndex=1))

        #carla本车信息获取
        #在当前车道上已行驶距离 单位 米
        carla_lane_pos=traci.vehicle.getLanePosition(carla_id)
        carla_speed=traci.vehicle.getSpeed(carla_id)
        carla_lane=traci.vehicle.getLaneIndex(carla_id)
        carla_road_id=traci.vehicle.getRoadID(carla_id)
        #print(carla_id,": getLanePosition: ",carla_lane_pos,"; getSpeed: ",carla_speed)

        #if carla_road_id==simulation_route[0]:
        #    del(simulation_route[0])

        #加载路网文件
        net= sumolib.net.readNet(r'D:\Carla\Co-Simulation\Sumo\examples\net\net.net.xml', withInternal=True)    
        #print('next:',traci.vehicle.getDrivingDistance(vehID=carla_id,edgeID=list(net.getEdge(carla_road_id).getOutgoing().keys())[0].getID(),pos=1,laneIndex=1))
        #获取道路长度 单位米
        #print(carla_id,": getLength: ",net.getEdge(carla_road_id).getLength())

        #是否生成车辆的标记
        add_check=False

        #previous_flow 0分流 1合流 2交织
        traffic_flow=-1
        ramp_area=-1
        generate=previous_generate

        if carla_road_id=='-43' and state==0:
            state=1
        elif carla_road_id=='5' and state==1:
            state=2

        #判断edgeID是否变化 且 速度要过关 m/s
        if last_edge_id!=carla_road_id and carla_speed>10:

            #是否900米交织区后半段已经生成新车的标记，初次道路EDGEID变化，必然未生成
            endofEdgeGenerated=False
            #是否900米交织区中部段已经生成新车的标记，初次道路EDGEID变化，必然未生成
            midofEdgeGenerated=False
            #是否交织区路段
            if 880<net.getEdge(carla_road_id).getLength()<920:
                #print('wave')
                #前50m
                if 60<carla_lane_pos<65 and previous_generate==0:
                    if state==1:#监管者模式
                        winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_auto.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                    elif state==2:#乘坐者模式
                        winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_play.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                elif 150<carla_lane_pos<300 and previous_generate==0:
                    #winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\front.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                    print('behind on-ramp')
                    add_check=True
                    #previous_flow 0分流 1合流 2交织
                    traffic_flow=0 if previous_flow[1]==1 else 1
                    previous_flow[1]=traffic_flow
                    ramp_area=1
                    generate=1
                elif previous_generate==2:
                    last_edge_id=carla_road_id
                    return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,generate]
            #是否两交织区中间短路段
            #elif 130<net.getEdge(carla_road_id).getLength()<150:
            #    print('front on-ramp')
            #    #add_check=True
            #是否四边的弯道路段
            #elif 1200<net.getEdge(carla_road_id).getLength()<1700:
            #    #仅后300m路段需要生成新车
            #    if (net.getEdge(carla_road_id).getLength()-carla_lane_pos)<300:
            #        print('front on-ramp')
                    #add_check=True
            #其他路段暂时不考虑主动式生成新车
            else:
                return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,generate]
        #如果EdgeID不发生变化 且 在900米交织区路段 且 速度过关 m/s
        elif last_edge_id==carla_road_id and 880<net.getEdge(carla_road_id).getLength()<920 and carla_speed>10:
            #判断是否中间路段 且 是否生成过标记为False 即未生成过
            if 60<carla_lane_pos<65 and previous_generate==2:
                if state==1:#监管者模式
                    winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_auto.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                elif state==2:#乘坐者模式
                    winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_play.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
            elif 350<carla_lane_pos<450 and midofEdgeGenerated==False and previous_generate==2:
                #winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\mid.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                print('wave area')
                #标记已生成车辆
                midofEdgeGenerated=True
                add_check=True
                #previous_flow 0分流 1合流 2交织
                traffic_flow=0 if previous_flow[2]==1 else 1
                previous_flow[2]=traffic_flow
                ramp_area=2
                generate=0
            elif 400<carla_lane_pos<405 and previous_generate==1:
                if state==1:#监管者模式
                    winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_auto.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                elif state==2:#乘坐者模式
                    winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\woman_play.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
            #判断是否后300m路段 且 是否生成过标记为False 即未生成过
            elif (net.getEdge(carla_road_id).getLength()-carla_lane_pos)<400 and endofEdgeGenerated==False and previous_generate==1:
                #winsound.PlaySound(r'D:\Carla\PythonAPI\examples\sounds\end.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
                print('front off-ramp')
                #标记已生成车辆
                endofEdgeGenerated=True
                add_check=True
                #previous_flow 0分流 1合流 2交织
                traffic_flow=0 if previous_flow[0]==1 else 1
                previous_flow[0]=traffic_flow
                ramp_area=0
                generate=2
            #未满足条件
            else:
                return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,generate]
        else:
            return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,generate]

        #如果有生成sumo新车标记  
        if add_check==True:
            #print('next:',traci.edge.getIDList())
            #print('next:',net.getEdge(carla_road_id))
            #print('next:',net.getEdges(withInternal=False))
            #print('next:',net.getEdge(carla_road_id).getOutgoing())
            #print('next:',type(net.getEdge(carla_road_id).getOutgoing()))
            #print('next:',net.getEdge(carla_road_id).getOutgoing().keys())
            #print('next:',list(net.getEdge(carla_road_id).getOutgoing().keys()))
            #print('next:',type(list(net.getEdge(carla_road_id).getOutgoing().keys())[0]))
            #接下来的edgeID
            #print('next:',list(net.getEdge(carla_road_id).getOutgoing().keys())[0].getID())

            #赋予sumo车当前edge和下一个edge之间的route
            if len(list(net.getEdge(carla_road_id).getOutgoing().keys()))>1:
                if carla_lane==0 or carla_lane==1:
                    traci.route.add(sumo_id+"_route", [carla_road_id,list(net.getEdge(carla_road_id).getOutgoing().keys())[0].getID()])
                else:
                    traci.route.add(sumo_id+"_route", [carla_road_id,list(net.getEdge(carla_road_id).getOutgoing().keys())[1].getID()])
            else:
                traci.route.add(sumo_id+"_route", [carla_road_id,list(net.getEdge(carla_road_id).getOutgoing().keys())[0].getID()])

            #确定生成车的车道
            #规则为：道路前进方向右侧第一条道为0（可能是右侧应急车道）
            sumo_lane=carla_lane
            #仅在900米交织区路段存在应急车道变可行驶车道的情况
            if 880<net.getEdge(carla_road_id).getLength()<920:
                
                if carla_lane==0:
                    sumo_lane=1
                elif carla_lane==3:
                    sumo_lane=2
                else:
                    #sumo_lane=random.choice([carla_lane-1,carla_lane+1])
                    sumo_lane=carla_lane-1
            else:
                if carla_lane==1 or carla_lane==3:
                    sumo_lane=2
                else:
                    #sumo_lane=random.choice([1,3])
                    sumo_lane=1
            
            #弃用，getLaneIndex更好；getLaneID的格式是 edgeID_laneID: -14_1，所以需要分割提取
            #carla_lane=traci.vehicle.getLaneID(carla_id).split("_")[-1]
            #print('Carla Lane: ',carla_lane,type(carla_lane))
            
            #simulation_next_edge_id=simulation_route[0]
            #if simulation_next_edge_id in [-6,21,-33,36,-34,19,-14,9,-13,28,-20,37,-27,18,-47,10]:
            #    print()
            #elif simulation_next_edge_id in [40,7,30,26,38,8,31,3,5,12,2,46,-1,-24,-41,-45,-22,-43,-23,-15,-32,-17,-29,-35]:
            #    print()
            #else:
            #    print()

            #生成sumo车辆
            #print('sumo_lane: ',sumo_lane,type(sumo_lane))
            sumo_main_depart_position=traci.vehicle.getLanePosition(carla_id)-random.uniform(5,10)
            depart_speed=carla_speed+random.uniform(5,7)
            #type_id_kind: 0=小型车 1=轿车 2=SUV/MPV 3=大型车/货车
            type_id_kind=random.choice([0,1,2,3])
            total_vehicle=['vehicle.micro.microlino','vehicle.mini.cooper_s','vehicle.nissan.micra','vehicle.audi.a2','vehicle.seat.leon',
                        'vehicle.lincoln.mkz_2017','vehicle.tesla.model3','vehicle.ford.mustang','vehicle.nissan.patrol','vehicle.jeep.wrangler_rubicon',
                        'vehicle.bmw.grandtourer','vehicle.carlamotors.carlacola','vehicle.tesla.cybertruck','vehicle.volkswagen.t2','vehicle.audi.tt',
                        'vehicle.dodge.charger_police','vehicle.chevrolet.impala','vehicle.audi.etron','vehicle.toyota.prius','vehicle.citroen.c3','vehicle.mercedes.coupe']

            type_id='vehicle.tesla.model3'
            if type_id_kind==0:
                type_id=random.choice(['vehicle.micro.microlino','vehicle.mini.cooper_s','vehicle.nissan.micra','vehicle.audi.a2'])
            elif type_id_kind==1:
                type_id=random.choice(['vehicle.seat.leon','vehicle.lincoln.mkz_2017','vehicle.tesla.model3','vehicle.ford.mustang'])
            elif type_id_kind==2:
                type_id=random.choice(['vehicle.nissan.patrol','vehicle.jeep.wrangler_rubicon','vehicle.bmw.grandtourer'])
            elif type_id_kind==3:
                type_id=random.choice(['vehicle.carlamotors.carlacola','vehicle.tesla.cybertruck','vehicle.volkswagen.t2'])

            try:
                traci.vehicle.add(vehID=sumo_id, routeID=sumo_id+"_route", typeID=type_id, depart='now', departLane=sumo_lane, departPos=sumo_main_depart_position, departSpeed=depart_speed, 
                                    arrivalLane=carla_lane, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
            except:
                print('sournding vehicle error')

            #low flow
            if traffic_flow==0:
                if carla_lane==1 or carla_lane==2:
                    if sumo_lane==carla_lane-1:
                        try:
                            traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                        except:
                            print('sournding vehicle error')
                    elif sumo_lane==carla_lane+1:
                        try:
                            traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                        except:
                            print('sournding vehicle error')
                elif carla_lane==0:
                    try:
                        traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                    except:
                        print('sournding vehicle error')
                elif carla_lane==3:
                    try:
                        traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                    except:
                        print('sournding vehicle error')
            #high flow
            elif traffic_flow==1:
                if carla_lane==1 or carla_lane==2:
                    if sumo_lane==carla_lane-1:
                        try:
                            traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+3', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+4', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                        except:
                            print('sournding vehicle error')
                    elif sumo_lane==carla_lane+1:
                        try:
                            traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed,
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+3', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(15,40), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                            traci.vehicle.add(vehID=sumo_id+'+4', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                        except:
                            print('sournding vehicle error')
                elif carla_lane==0:
                    try:
                        traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                                arrivalLane=carla_lane+1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                        traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+2, departPos=traci.vehicle.getLanePosition(carla_id), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane+2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                        traci.vehicle.add(vehID=sumo_id+'+3', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+2, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane+2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                    
                        traci.vehicle.add(vehID=sumo_id+'+4', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane+2, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(10,40), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane+2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                    except:
                        print('sournding vehicle error')
                elif carla_lane==3:
                    try:
                        traci.vehicle.add(vehID=sumo_id+'+1', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-1, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane-1, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                        traci.vehicle.add(vehID=sumo_id+'+2', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-2, departPos=traci.vehicle.getLanePosition(carla_id), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane-2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                        traci.vehicle.add(vehID=sumo_id+'+3', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-2, departPos=traci.vehicle.getLanePosition(carla_id)+random.uniform(90,110), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane-2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)

                        traci.vehicle.add(vehID=sumo_id+'+4', routeID=sumo_id+"_route", typeID=random.choice(total_vehicle), depart='now', departLane=carla_lane-2, departPos=traci.vehicle.getLanePosition(carla_id)-random.uniform(10,40), departSpeed=carla_speed, 
                                            arrivalLane=carla_lane-2, arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=1, personNumber=0)
                    except:
                        print('sournding vehicle error')
            #旧方法addLegacy
            #traci.vehicle.addLegacy(vehID=sumo_id,routeID=sumo_id+"_route",typeID='vehicle.tesla.model3', depart=0, lane=sumo_lane, pos=traci.vehicle.getLanePosition(carla_id)+20, speed=carla_speed+3)
            print('~'+str(current_time)+'#'+str(number)+'#'+str(round(number*0.05,2))+'#'+str(state)+'#'+str(traffic_flow)+'#'+str(ramp_area)+'#'+str(carla_lane)+'#'+str(sumo_id)+'#'+str(sumo_lane)+'#'+str(sumo_main_depart_position)+'#'+str(depart_speed)+'#'+str(type_id_kind),file=output_file)
            print('Added Vehicle Success')
            
            #回传lastEdgeID到run_sync.py
            last_edge_id=carla_road_id
            
            #测试生成车辆数量是否正确
            #with open(r'C:\Users\Eddy\Desktop\output_exp1.txt', 'a') as f:
            #    f.write(str(carla_road_id)+sumo_id)
            #    f.writelines("\n")

            #需要进一步确认，可能未成功生成，位置或速度未达标
        return [last_edge_id,midofEdgeGenerated,endofEdgeGenerated,simulation_route,previous_flow,state,generate]

    def synchronize_vehicle(self, vehicle_id, transform, signals=None):
        """``
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param signals: new vehicle signals.
            :return: True if successfully updated. Otherwise, False.
        """
        loc_x, loc_y = transform.location.x, transform.location.y
        yaw = transform.rotation.yaw

        traci.vehicle.moveToXY(vehicle_id, "", 0, loc_x, loc_y, angle=yaw, keepRoute=2)
        if signals is not None: 
            traci.vehicle.setSignals(vehicle_id, signals)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param tl_id: id of the traffic light to be updated (logic id, link index).
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        self.traffic_light_manager.set_state(landmark_id, state)

    def tick(self):
        """
        Tick to sumo simulation.
        """
        traci.simulationStep()
        self.traffic_light_manager.tick()

        # Update data structures for the current frame.
        self.spawned_actors = set(traci.simulation.getDepartedIDList())
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())

    @staticmethod
    def close():
        """
        Closes traci client.
        """
        traci.close()
