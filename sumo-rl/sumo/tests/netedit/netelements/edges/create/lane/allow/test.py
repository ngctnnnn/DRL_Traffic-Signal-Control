#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    test.py
# @author  Pablo Alvarez Lopez
# @date    2016-11-25

# import common functions for netedit tests
import os
import sys

testRoot = os.path.join(os.environ.get('SUMO_HOME', '.'), 'tests')
neteditTestRoot = os.path.join(
    os.environ.get('TEXTTEST_HOME', testRoot), 'netedit')
sys.path.append(neteditTestRoot)
import neteditTestFunctions as netedit  # noqa

# Open netedit
neteditProcess, referencePosition = netedit.setupAndStart(neteditTestRoot, ['--new'])

# Change to create mode
netedit.createEdgeMode()

# set attribute
netedit.changeDefaultValue(netedit.attrs.edge.createLane.allow, "dummy")

# Create two nodes
netedit.leftClick(referencePosition, 80, 100)
netedit.leftClick(referencePosition, 510, 100)

# set attribute
netedit.changeDefaultValue(netedit.attrs.edge.createLane.allow, "pedestrian bus")

# Create two nodes
netedit.leftClick(referencePosition, 80, 175)
netedit.leftClick(referencePosition, 500, 175)

# set attribute
netedit.changeDefaultValue(netedit.attrs.edge.createLane.allow, "all")

# Create two nodes
netedit.leftClick(referencePosition, 80, 250)
netedit.leftClick(referencePosition, 500, 250)

# set attribute
netedit.changeDefaultAllowDisallowValue(netedit.attrs.edge.createLane.allowButton)

# Create two nodes
netedit.leftClick(referencePosition, 80, 325)
netedit.leftClick(referencePosition, 500, 325)

# Check undo and redo
netedit.undo(referencePosition, 3)
netedit.redo(referencePosition, 3)

# rebuild network
netedit.rebuildNetwork()

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
