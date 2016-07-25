from Box2D import *
import cairo
from collections import namedtuple
import gc
import itertools
import numpy as np
import os

N_LINKS = 3
N_BLOCKS = 2
SCALE = 3
N_INTERP = 20

class Datum(namedtuple("Datum", ["features", "init", "goal", "demonstration", "config"])):
    def inject_state_features(self, state):
        return self.features
        #return np.concatenate((self.features, state))

Config = namedtuple("Config", ["init", "goal","len_links", "pos_blocks"])

BASE_FIXTURE_DEF = b2FixtureDef(
        shape=b2PolygonShape(box=(SCALE * 0.03, SCALE * 0.03), density=1.))
BLOCK_FIXTURE_DEF = b2FixtureDef(
        shape=b2PolygonShape(box=(SCALE * 0.01, SCALE * 0.01), density=1.))
LINK_LENS = list(np.arange(0.02, 0.08, 0.005) * 2)
LINK_FIXTURE_DEFS = [
    b2FixtureDef(shape=b2PolygonShape(box=(SCALE * 0.005, SCALE * l / 2)),
                 density=1.)
    for l in LINK_LENS
]

ANGLES = np.arange(-1, 1, 0.2) * np.pi
POSES = list(itertools.product(ANGLES, repeat=3))

#@profile
def load_batch(n_batch):
    data = []
    while len(data) < n_batch:
        config = sample_config()
        #midpoint = list((np.random.random(N_LINKS) * 2 - 1) * np.pi)
        i_midpoint = np.random.randint(len(POSES))
        midpoint = POSES[i_midpoint]
        direct_path = [config.init, config.goal]
        wp_path = [config.init, midpoint, config.goal]
        if verify(config, direct_path) is not None:
            continue
        if verify(config, wp_path) is None:
            continue
        datum = Datum(np.zeros(N_BLOCKS), config.init, config.goal, [0, i_midpoint], config)
        data.append(datum)

        #visualize(data[0].config, verify(config, wp_path))
        #exit()
    return data

def evaluate(path, datum):
    true_path = [datum.init, POSES[np.argmax(path[1])], datum.goal]
    succ = verify(datum.config, true_path) is not None
    return 1. if succ else 0.

def visualize(path, datum):
    true_path = [datum.init, POSES[np.argmax(path[1])], datum.goal]
    interp_path = interpolate(true_path)
    animate(interp_path, datum.config)

def animate(path, config):
    if os.path.exists("out.mp4"):
        os.remove("out.mp4")
    frames = []
    t = 0
    for dof in path:
        world = build_world(config, dof)
        frame = "%d.png" % t
        assert not os.path.exists(frame)
        render(world, frame)
        frames.append(frame)
        #reset_world(world)
        t += 1
    os.system("ffmpeg -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4")
    for frame in frames:
        os.remove(frame)

# helpers

def sample_config():
    #link_lengths = list((np.random.random(N_LINKS) * 0.04 + 0.02) * SCALE)
    #link_lengths[0] += 0.02
    link_len_ids = np.random.randint(len(LINK_LENS) - 4, size=N_LINKS)
    link_len_ids[0] += 4
    link_lengths = [LINK_LENS[i] for i in link_len_ids]

    init_dof = list((np.random.random(N_LINKS) * 2 - 1) * np.pi)
    goal_dof = [0] * N_LINKS

    use_blocks = np.random.random(size=(N_LINKS, N_BLOCKS)) < 0.5
    block_angles = (np.random.random(size=(N_LINKS, N_BLOCKS)) * 2 - 1) * np.pi

    blocks = []
    running_scale = 0
    for i in range(N_LINKS):
        r = running_scale + link_lengths[i] / 4
        #for j in range(2):
        for j in range(N_BLOCKS):
            if not use_blocks[i][j]:
                continue
            x = np.cos(block_angles[i][j]) * r
            y = np.sin(block_angles[i][j]) * r
            blocks.append((x, y))
        running_scale = r + link_lengths[i] / 4

    return Config(init_dof, goal_dof, link_lengths, blocks)

def build_world(config, dof):
    world = b2World(gravity=(0, 0))
    assert world.bodyCount == 0

    # anchor
    #if len(shared_anchors) == 0:
    base = world.CreateStaticBody(position=(0, 0))
    base.CreateFixture(BASE_FIXTURE_DEF)
    #anchor.CreatePolygonFixture(box=(SCALE * 0.03, SCALE * 0.03))
        #shared_anchors.append(anchor)
    #else:
    #    anchor = shared_anchors[0]
    prev = base
    prev_len = 0

    # links
    links = []
    for i_link in range(N_LINKS):
        link = world.CreateDynamicBody()
        #link.CreatePolygonFixture(box=(SCALE * 0.005,
        #                               SCALE * config.len_links[i_link] / 2),
        #                          density=1,
        #                          friction=0.3)
        link.CreateFixture(LINK_FIXTURE_DEFS[LINK_LENS.index(config.len_links[i_link])])
        link_len = SCALE * config.len_links[i_link] / 2
        joint = world.CreateRevoluteJoint(
                bodyA=prev,
                bodyB=link,
                localAnchorA=(0, -prev_len),
                localAnchorB=(0, link_len),
                #enableMotor=True,
                #maxMotorTorque=400,
                #motorSpeed=1,
                #enableLimit=False
                )
        if i_link == 0:
            joint.enableMotor = True
            joint.motorSpeed = 1.0
            joint.maxMotorTorque = 400
        prev = link
        prev_len = link_len
        links.append(link)

    parent_angle = 0
    parent = base
    parent_len = 0
    for i_link in range(N_LINKS):
        link = links[i_link]
        angle = dof[i_link]
        #angle = 0
        pos = parent.GetWorldPoint((0, -parent_len))
        link.angle = parent_angle + angle
        link_len = SCALE * config.len_links[i_link] / 2
        new_pos = link.GetWorldPoint((0, link_len))
        link.position += pos - new_pos 
        parent_angle = link.angle
        parent = link
        parent_len = link_len

    # blocks
    blocks = []
    for block_pos in config.pos_blocks:
        block_body = world.CreateStaticBody(
                position=(2 * SCALE * block_pos[0], 2 * SCALE * block_pos[1]))
        #block_body.CreatePolygonFixture(
        #        box=(SCALE * 0.01, SCALE * 0.01),
        #        isSensor=True)
        block_body.CreateFixture(BLOCK_FIXTURE_DEF)
        blocks.append(block_body)

    world.Step(0.01, 6, 2)
    return world

def simulate(world, n_frames):
    if os.path.exists("out.mp4"):
        os.remove("out.mp4")
    frames = []
    for t in range(n_frames):
        world.Step(0.01, 6, 2)
        frame = "%d.png" % t
        assert not os.path.exists(frame)
        render(world, frame)
        frames.append(frame)
    os.system("ffmpeg -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4")
    for frame in frames:
        os.remove(frame)

def render(world, dest):
    WIDTH = 500
    HEIGHT = 500
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1., 1., 1.)
    ctx.paint()
    ctx.set_source_rgb(0., 0., 0.)
    ctx.set_line_width(0.01)

    def reset_transform():
        ctx.identity_matrix()
        ctx.scale(WIDTH / SCALE, HEIGHT / SCALE)
        ctx.translate(0.5 * SCALE, 0.5 * SCALE)
        ctx.scale(1., -1.)

    for body in world.bodies:
        reset_transform()
        transform = body.transform
        ctx.translate(*transform.position)
        ctx.rotate(transform.angle)
        for fixture in body.fixtures:
            shape = fixture.shape
            assert type(shape) == b2PolygonShape
            vertices = shape.vertices
            ctx.move_to(*vertices[-1])
            for vertex in vertices:
                ctx.line_to(*vertex)
            ctx.close_path()
            ctx.stroke()
    surf.write_to_png(dest)

#@profile
def interpolate(path):
    interp_path = []
    for i in range(1, len(path)):
        fr = np.asarray(path[i-1])
        to = np.asarray(path[i])
        for j in range(N_INTERP):
            d = (1. * j / N_INTERP)
            interp = list((1 - d) * fr + d * to)
            interp_path.append(interp)
    interp_path.append(path[-1])
    return interp_path

def verify(config, path):
    interp_path = interpolate(path)
    for dof in interp_path:
        world = build_world(config, dof)
        #if np.random.random() < 0.1:
        #    return None
        for contact in world.contacts:
            if contact.touching:
                return None
    return interp_path
