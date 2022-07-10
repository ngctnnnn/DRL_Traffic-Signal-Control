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
neteditProcess, referencePosition = netedit.setupAndStart(neteditTestRoot)

# recompute
netedit.rebuildNetwork()

# force save additionals
netedit.forceSaveAdditionals()

# toggle select lanes
netedit.changeEditMode(netedit.attrs.modes.network.selectLane)

# go to select mode
netedit.selectMode()

# select first lane
netedit.leftClick(referencePosition, 250, 155)

# select second lane
netedit.leftClick(referencePosition, 250, 95)

# go to inspect mode
netedit.inspectMode()

# inspect lane
netedit.leftClick(referencePosition, 250, 95)

# Change parameter 0 with a non valid value (empty speed)
netedit.modifyAttribute(netedit.attrs.lane.inspectSelection.speed, "", True)

# Change parameter 0 with a non valid value (dummy speed)
netedit.modifyAttribute(netedit.attrs.lane.inspectSelection.speed, "dummySpeed", True)

# Change parameter 0 with a non valid value (negative speed)
netedit.modifyAttribute(netedit.attrs.lane.inspectSelection.speed, "-13", True)

# Change parameter 0 with a valid value
netedit.modifyAttribute(netedit.attrs.lane.inspectSelection.speed, "120.5", True)

# recompute
netedit.rebuildNetwork()

# Check undo
netedit.undo(referencePosition, 1)

# recompute
netedit.rebuildNetwork()

# Check redo
netedit.redo(referencePosition, 1)

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
