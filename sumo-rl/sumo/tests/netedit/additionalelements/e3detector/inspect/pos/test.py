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
neteditProcess, referencePosition = netedit.setupAndStart(neteditTestRoot, ['--gui-testing-debug-gl'])

# apply zoom
netedit.setZoom("25", "25", "25")

# go to additional mode
netedit.additionalMode()

# select E3
netedit.changeElement("e3Detector")

# create E3
netedit.leftClick(referencePosition, 250, 110)

# select entry detector
netedit.changeElement("detEntry")

# Create Entry detector E3 (for saving)
netedit.leftClick(referencePosition, 250, 110)
netedit.leftClick(referencePosition, 250, 240)

# select entry detector
netedit.changeElement("detExit")

# Create Exit detector E3 (for saving)
netedit.leftClick(referencePosition, 250, 110)
netedit.leftClick(referencePosition, 250, 420)

# go to inspect mode
netedit.inspectMode()

# inspect first E3
netedit.leftClick(referencePosition, 250, 110)

# Change parameter position with a non valid value (dummy position)
netedit.modifyAttribute(netedit.attrs.E3.inspect.pos, "dummy position", False)

# Change parameter position with a non valid value (empty)
netedit.modifyAttribute(netedit.attrs.E3.inspect.pos, "", False)

# Change parameter position with a valid value (different position)
netedit.modifyAttribute(netedit.attrs.E3.inspect.pos, "25, 25", False)

# Check undos and redos
netedit.undo(referencePosition, 5)
netedit.redo(referencePosition, 5)

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
