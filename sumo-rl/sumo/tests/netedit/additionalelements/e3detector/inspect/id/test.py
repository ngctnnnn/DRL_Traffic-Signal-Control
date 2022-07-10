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

# apply zoom
netedit.setZoom("25", "25", "25")

# go to additional mode
netedit.additionalMode()

# select E3
netedit.changeElement("e3Detector")

# create E3 1
netedit.leftClick(referencePosition, 250, 110)

# create E3 2 (for check duplicated ID)
netedit.leftClick(referencePosition, 450, 110)

# select entry detector
netedit.changeElement("detEntry")

# Create Entry detector E3 (for saving)
netedit.leftClick(referencePosition, 250, 110)
netedit.leftClick(referencePosition, 250, 250)
netedit.leftClick(referencePosition, 450, 110)
netedit.leftClick(referencePosition, 450, 250)

# select entry detector
netedit.changeElement("detExit")

# Create Exit detector E3 (for saving)
netedit.leftClick(referencePosition, 250, 110)
netedit.leftClick(referencePosition, 250, 420)
netedit.leftClick(referencePosition, 450, 85)
netedit.leftClick(referencePosition, 450, 420)

# go to inspect mode
netedit.inspectMode()

# inspect first E3
netedit.leftClick(referencePosition, 250, 110)

# Change parameter id with a non valid value (Duplicated ID)
netedit.modifyAttribute(netedit.attrs.E3.inspect.id, "e3_1", False)

# Change parameter id with a non valid value (invalid characters)
netedit.modifyAttribute(netedit.attrs.E3.inspect.id, ";;;;;;;;;;;;;;;;;", False)

# Change parameter id with a valid value (with spaces)
netedit.modifyAttribute(netedit.attrs.E3.inspect.id, "Id with spaces", False)

# Change parameter id with a valid value
netedit.modifyAttribute(netedit.attrs.E3.inspect.id, "correctID", False)

# Check undos and redos
netedit.undo(referencePosition, 8)
netedit.redo(referencePosition, 8)

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
