#!/usr/bin/env python2

from Box2D import *

def main():

    shape = b2PolygonShape(box=(1, 1))
    fixture_def = b2FixtureDef(shape=shape)
    for i in xrange(99999999):
        world = b2World(gravity=(0, 0))
        body = world.CreateStaticBody(position=(0, 0))

        #shape = b2PolygonShape(box=(1, 1))
        fixture = body.CreateFixture(fixture_def)
        body.DestroyFixture(fixture)
        #del fixture

        #fixture = body.CreatePolygonFixture(box=(1, 1))

        #print dir(world)
        #world.ClearForces()

        #for b in world.bodies:
        #    world.DestroyBody(b)

main()
