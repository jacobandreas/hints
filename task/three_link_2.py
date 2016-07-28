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

#ANGLES = np.arange(-1, 1, 0.2) * np.pi
#POSES = list(itertools.product(ANGLES, repeat=3))
#POSES = [(0, 0, 0), (0, 2.5, -2.5)]
POSES = [(np.pi / 2, 2.5, -2.5), (-np.pi / 2, 2.5, -2.5),
         (np.pi / 2, -2.5, 2.5), (-np.pi / 2, -2.5, 2.5)]

all_samples = []
batch_uses = [0]

#@profile
def load_batch(n_batch):
    if batch_uses[0] == 10 or len(all_samples) == 0:
        batch_uses[0] = 0
        data = []
        while len(data) < n_batch:
            config = sample_config()
            i_midpoint = np.random.randint(len(POSES))
            midpoint = POSES[i_midpoint]
            direct_path = [config.init, config.goal]
            wp_path = [config.init, midpoint, config.goal]
            wp_path = zip(*[wrap_angle(pp) for pp in zip(*wp_path)])

            if verify(config, direct_path) is not None:
                continue
            if verify(config, wp_path) is None:
                continue

            block_features = np.zeros((N_BLOCKS * N_LINKS, 2))
            for i_block, block in enumerate(config.pos_blocks):
                block_features[i_block, :] = block
            features = np.concatenate(
                    (block_features.ravel(), config.len_links, config.init, config.goal))

            #demonstration = [config.init, midpoint, config.goal]
            demonstration = [0, i_midpoint]
            datum = Datum(features, config.init, config.goal, demonstration, config)

            #fake_demo = np.zeros(len(POSES))
            #fake_demo[i_midpoint] = 1
            #fake_demo = [None, fake_demo]
            #if evaluate(fake_demo, datum) != 1:
            #    print verify(config, wp_path)
            #    print evaluate(fake_demo, datum)
            #    print ">>", demonstration[1]
            #    visualize(demonstration, datum)
            #    exit()
            #assert evaluate(fake_demo, datum) == 1.

            data.append(datum)

        for d in data: all_samples.append(d)
        if len(all_samples) > 10 * n_batch:
            del all_samples[:n_batch]

    batch_uses[0] += 1
    batch_ids = np.random.randint(len(all_samples), size=n_batch)
    batch_data = [all_samples[i] for i in batch_ids]
    return batch_data

def evaluate(path, datum):
    #print path
    assert len(path[1]) == len(POSES)
    true_path = [datum.init, POSES[np.argmax(path[1])], datum.goal]
    #true_path = path
    true_path = zip(*[wrap_angle(pp) for pp in zip(*true_path)])
    succ = verify(datum.config, true_path) is not None
    return 1. if succ else 0.

def visualize(path, datum, name="vis"):
    assert len(path[1]) == len(POSES)
    true_path = [datum.init, POSES[np.argmax(path[1])], datum.goal]
    true_path = zip(*[wrap_angle(pp) for pp in zip(*true_path)])
    interp_path = interpolate(true_path)
    interp_path = zip(*[wrap_angle(pp) for pp in zip(*interp_path)])
    animate(interp_path, datum.config, name)

def animate(path, config, name):
    fname = "%s.mp4" % name
    if os.path.exists(fname):
        os.remove(fname)
    frames = []
    t = 0
    for dof in path:
        world = build_world(config, dof)
        frame = "%d.png" % t
        assert not os.path.exists(frame)
        render(world, frame)
        frames.append(frame)
        t += 1
    os.system("ffmpeg -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p " + fname)
    for frame in frames:
        os.remove(frame)

# helpers

def sample_config():
    #link_lengths = list((np.random.random(N_LINKS) * 0.04 + 0.02) * SCALE)
    #link_lengths[0] += 0.02
    link_len_ids = np.random.randint(len(LINK_LENS) - 4, size=N_LINKS)
    link_len_ids[0] += 4
    link_lengths = [LINK_LENS[i] for i in link_len_ids]

    #init_dof = [np.pi, 0, 0]
    init_dof = list((np.random.random(N_LINKS) * 2 - 1) * np.pi)
    goal_dof = [0] * N_LINKS

    use_blocks = np.random.random(size=(N_LINKS, N_BLOCKS)) < 0.5
    block_angles = (np.random.random(size=(N_LINKS, N_BLOCKS)) * 2 - 1) * np.pi

    blocks = []
    #running_scale = link_lengths[0] / 2
    running_scale = 0
    for i in range(0, N_LINKS):
        r = running_scale + link_lengths[i] / 4
        #for j in range(2):
        for j in range(N_BLOCKS):
            if not use_blocks[i][j]:
                continue
            x = np.cos(block_angles[i][j]) * r
            y = np.sin(block_angles[i][j]) * r
            blocks.append((x, y))
        running_scale = r + link_lengths[i] / 4
    #r = link_lengths[0] / 4
    #angle = np.pi / 2
    #if np.random.random() < 0.5:
    #    angle *= -1
    #x = np.sin(angle) * r
    #y = np.cos(angle) * r
    #blocks = [(x, y)]

    return Config(init_dof, goal_dof, link_lengths, blocks)

def build_world(config, dof):
    world = b2World(gravity=(0, 0))
    assert world.bodyCount == 0

    # anchor
    base = world.CreateStaticBody(position=(0, 0))
    base.CreateFixture(BASE_FIXTURE_DEF)
    prev = base
    prev_len = 0

    # links
    links = []
    for i_link in range(N_LINKS):
        link = world.CreateDynamicBody()
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
        for contact in world.contacts:
            if contact.touching:
                return None
    return interp_path

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
