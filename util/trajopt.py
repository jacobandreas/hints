import json
import numpy as np
import openravepy
import os
import psutil
import time
import trajoptpy
from trajoptpy import check_traj

counter = [0]
env = [openravepy.Environment()]

def interp(begin, end, count):
    assert count >= 2
    out = [begin]
    step = 1. / (count - 1)
    for i_step in range(1, count):
        delta = step * i_step
        here = end * delta + begin * (1 - delta)
        out.append(here)
    return np.asarray(out)

def plan(init_dof, waypoints, goal_dof, env):
    robot = env[0].GetRobots()[0]

    robot.SetDOFValues(init_dof)
    #init_traj = np.concatenate((
    #        interp(init_dof, waypoint, 10), interp(waypoint, goal_dof, 10)))
    init_traj = waypoints
    #init_traj = interp(init_dof, goal_dof, 20)
    #init_traj = init_traj.tolist()
    request = {
        "basic_info" : {
            "n_steps" : len(init_traj),
            "manip" : "active",
            "start_fixed" : True
        },
        "costs" : [
            {
                "type" : "joint_vel",
                "params" : { "coeffs" : [1] }
            },
            {
                "type" : "collision",
                "params" : {
                    "coeffs" : [20],
                    "dist_pen" : [0.025]
                },
            }
        ],
        "constraints" : [
            {
                "type" : "joint",
                "params" : { "vals" : goal_dof }
            }
        ],
        "init_info" : {
            "type" : "given_traj",
            "data" : init_traj
        }
    }
    #s = json.dumps(request)
    #prob = trajoptpy.ConstructProblem(s, env)
    #result = trajoptpy.OptimizeProblem(prob)
    #traj = result.GetTraj()
    traj = init_traj
    if check_traj.traj_is_safe(traj, robot):
        return 1
    else:
        return 0

def cleanup():
    counter[0] += 1
    proc = psutil.Process()
    for pfile in proc.open_files():
        if ".openrave/3DOFRobot.traj.xml" in pfile.path:
            os.close(pfile.fd)

    for body in env[0].GetBodies():
        env[0].Remove(body)
    assert len(env[0].GetRobots()) == 0
    if counter[0] >= 10000:
        print "destroy"
        openravepy.RaveDestroy()
        openravepy.RaveInitialize()
        env[0] = openravepy.Environment()
        counter[0] = 0
        import gc
        gc.collect()
        #env[0].Destroy()
        #env[0] = None


def verify(datum, waypoints):
    env[0].Load(datum.env_path)
    try:
        return np.random.random() < 0.1
    finally:
        cleanup()
    try:
        robot = env[0].GetRobots()[0]
        robot.SetDOFValues(datum.init)
        traj = [datum.init] + list(waypoints) + [datum.goal]
        return 1 if check_traj.traj_is_safe(traj, robot) else 0
    except Exception as e:
        print "in verify:" + str(e)
        return 0

def vis(datum, waypoints):
    env.Load(datum.env_path)
    env.SetViewer("qtcoin")
    print "viewing"
    try:
        robot = env.GetRobots()[0]
        traj = [datum.init] + waypoints + [datum.goal]
        for point in traj:
            robot.SetDOFValues(point)
            time.sleep(2)
    finally:
        cleanup()

def solve(datum, waypoints):
    env.Load(datum.env_path)
    try:
        succ = plan(datum.init, waypoints, datum.goal, env)
        return succ
    except Exception as e:
        return 0
    finally:
        cleanup()

