#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2008-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @date    2010-10-17

from __future__ import absolute_import
from __future__ import print_function

import os
import subprocess
import sys
import shutil
sys.path.append(os.path.join(
    os.path.dirname(sys.argv[0]), '..', '..', '..', '..', '..', "tools"))
import traci  # noqa
import sumolib  # noqa

sumoBinary = sumolib.checkBinary('sumo')
netconvertBinary = sumolib.checkBinary('netconvert')

srcRoot = os.path.join(os.path.dirname(sys.argv[0]), "data")
roots = []
for root, dirs, files in os.walk(srcRoot):
    if "input_edges.edg.xml" in files:
        roots.append(root)

for root in sorted(roots):
    print("-- Test: %s" % root[len(srcRoot) + 1:])
    prefix = os.path.join(root, "input_")
    sys.stdout.flush()
    subprocess.call([netconvertBinary, "-n", prefix + "nodes.nod.xml", "-e", prefix + "edges.edg.xml",
                     "-x", prefix + "connections.con.xml", "-o", "./input_net.net.xml"], stdout=sys.stdout)
    sys.stdout.flush()
    shutil.copy(prefix + "routes.rou.xml", "./input_routes.rou.xml")
    shutil.copy(prefix + "additional.add.xml", "./input_additional.add.xml")

    traci.start([sumoBinary, "-c", "sumo.sumocfg"])
    traci.simulationStep()
    lanes = traci.vehicle.getBestLanes("0")
    sys.stdout.flush()
    for l in lanes:
        print("lane %s:" % (l[0]))
        print("  length: %s" % (l[1]))
        print("  offset: %s" % (l[3]))
        print("  allowsContinuation: %s" % (l[4]))
        print("  over: %s" % (l[5],))
    traci.close()
    sys.stdout.flush()

    with open(os.path.join(root, "expected.txt")) as fdi:
        for i, l in enumerate(lanes):
            vals = fdi.readline().strip().split()
            length = float(vals[0])
            if abs(float(l[1]) - length) < 100:
                print("lane %s ok" % i)
            else:
                print("lane %s mismatches" % i)
    print("-" * 70)
    print("")
