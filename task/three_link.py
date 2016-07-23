from util import trajopt

from collections import namedtuple
import numpy as np
import openravepy
import os
import psutil
import shutil
import xmltodict

PATH_LEN = 10

Task = namedtuple("Task", ["env_path", "init_dof", "goal_dof"])

class Datum(namedtuple("Datum", ["features", "init", "goal", "demonstration", "env_path"])):
    def inject_state_features(self, state):
        return np.concatenate((self.features, state))

robot_template_f = open("data/templates/three_link_robot.xml")
robot_template_str = "".join(robot_template_f.readlines())
robot_template_f.close()

block_template_f = open("data/templates/block.xml")
block_template_str = "".join(block_template_f.readlines())
block_template_f.close()

def build(dest):
    #link_lengths = [0.13, 0.08, 0.06]
    #init_dof = [np.pi - 0.1, 0.1, -0.1]
    #goal_dof = [0] * 3
    #block_angles = [np.pi / 2, np.pi / 2, -np.pi/2]

    #init_dof = [2 * np.pi - 0.1, 0.1, -0.1]
    #goal_dof = [np.pi, 0, 0]
    #block_angles = [3 * np.pi / 2, 3 * np.pi / 2, -3 * np.pi / 2]

    link_lengths = list(np.random.random(3) * 0.08 + 0.02)
    link_lengths[0] += 0.05
    init_dof = list((np.random.random(3) * 2 - 1) * np.pi)
    goal_dof = [0] * 3

    use_blocks = np.random.random(size=(3,2)) < 0.5
    block_angles = (np.random.random(size=(3,2)) * 2 - 1) * np.pi

    with open(dest, "w") as env_f:
        print >>env_f, "<Environment>"
        print >>env_f, _robot(*link_lengths)

        running_scale = 0
        for i in range(3):
            r = 2 * running_scale + link_lengths[i]
            running_scale += link_lengths[i]
            for j in range(2):
                if not use_blocks[i][j]:
                    continue
                x = np.cos(block_angles[i][j]) * r
                y = np.sin(block_angles[i][j]) * r
                print >>env_f, _block("block%d%d" % (i,j), 0.005, 0.005, 0.05, x, y, 0)

        print >>env_f, "</Environment>"
    return Task(dest, init_dof, goal_dof)

def load(id, env_path, init_dof, goal_dof, path):
    block_features = np.zeros((5, 3))
    robot_features = np.zeros((3,))

    with open(env_path) as config_f:
        config_doc = xmltodict.parse(config_f)

    blocks = config_doc["Environment"]["KinBody"]
    if not isinstance(blocks, list):
        blocks = [blocks]
    for i_block, block in enumerate(blocks):
        assert "block" in block["@name"]
        translation = block["Translation"]
        translation = [float(f) for f in translation.split()]
        translation = np.asarray(translation)
        block_features[i_block, :] = translation

    robot = config_doc["Environment"]["Robot"]["KinBody"]
    arms = [b for b in robot["Body"] if "Arm" in b["@name"]]
    for i_arm, arm in enumerate(arms):
        robot_features[i_arm] = float(arm["Geom"]["Extents"].split()[0])
    features = np.concatenate((block_features.ravel(),
            robot_features.ravel(), init_dof))

    #datum = PathDatum(id, init_dof, goal_dof, path, features, None,
    #        env_path)
    datum = Datum(features, init_dof, goal_dof, path, env_path)
    return datum

def evaluate(pred, datum):
    return trajopt.verify(datum, pred)

def visualize(pred, datum):
    return

def load_batch(n_batch):
    if os.path.exists("envs"):
        shutil.rmtree("envs")
    os.mkdir("envs")

    data = []
    i_example = 0
    while len(data) < n_batch:
        try:
            env_path = "envs/%d.xml" % i_example
            task = build(env_path)
            i_d = np.asarray(task.init_dof)
            g_d = np.asarray(task.goal_dof)

            #env = openravepy.Environment()
            #env.Load(env_path)
            #robot = env.GetRobots()[0]
            #robot.SetDOFValues(i_d)
            #params = openravepy.Planner.PlannerParameters()
            #params.SetRobotActiveJoints(robot)
            #params.SetGoalConfig(task.goal_dof)
            #params.SetExtraParameters("""
            #    <verifyinitialpath>0</verifyinitialpath>
            #    <_nmaxiterations>100</_nmaxiterations>
            #""") 
            #planner = openravepy.RaveCreatePlanner(env, "birrt")
            #planner.InitPlan(robot, params)
            #traj = openravepy.RaveCreateTrajectory(env, "")
            #planner.PlanPath(traj)
            #path = [traj.GetWaypoint(i).tolist()[:len(task.goal_dof)] for i in
            #        range(traj.GetNumWaypoints())]
            midpoint = list((np.random.random(3) * 2 - 1) * np.pi)
            path = list(trajopt.interp(np.asarray(task.init_dof),
                                       np.asarray(midpoint), PATH_LEN / 2)) + \
                   list(trajopt.interp(np.asarray(midpoint),
                                       np.asarray(task.goal_dof), PATH_LEN / 2))
            path = zip(*[wrap_angle(pp) for pp in zip(*path)])
            #if len(path) == 2:
            #    raise Exception("rrt failed")
            #npath = []
            #for i_np in range(PATH_LEN):
            #    idx = int(1. * i_np * len(path) / PATH_LEN)
            #    npath.append(path[idx])
            #path = npath

            datum = load(i_example, env_path, task.init_dof, task.goal_dof, path)
            if not trajopt.verify(datum, path):
                raise Exception("verify failed")

            easy_path = trajopt.interp(np.asarray(task.init_dof),
                    np.asarray(task.goal_dof), 8).tolist()
            if trajopt.verify(datum, easy_path):
                raise Exception("interp succeeded")

            data.append(datum)
            i_example += 1
        except Exception as e:
            pass
        finally:
            proc = psutil.Process()
            for pfile in proc.open_files():
                if ".openrave/3DOFRobot.traj.xml" in pfile.path:
                    os.close(pfile.fd)
            #for body in env.GetBodies():
            #    env.Remove(body)

    return data

def _robot(len1, len2, len3):
    robot_str = robot_template_str % (
            len1, len1,
             2 * len1 - 0.01, len2, len2,
             2 * len2 - 0.01, len3, len3)

    return robot_str

def _block(name, w, h, d, x, y, z):
    block_str = block_template_str % (
            name,
            x, y, z,
            w, h, d)

    return block_str

def wrap_angle(points):
    out = [points[0]]
    current = points[0]
    for point in points[1:]:
        # TODO more general
        candidates = [point + 2 * np.pi * i for i in range(-3, 3)]
        best = min(candidates, key=lambda c: abs(c - current))
        out.append(best)
        current = best
    assert len(points) == len(out)
    return out
