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

# go to additional mode
netedit.additionalMode()

# select E2
netedit.changeElement("e2Detector")

# set invalid jam threshold (dummy)
netedit.changeDefaultValue(netedit.attrs.E2.create.jamThreshold, "dummyJamTreshold")

# try to create E2 with invalid jam threshold
netedit.leftClick(referencePosition, 450, 220)

# set invalid jam threshold
netedit.changeDefaultValue(netedit.attrs.E2.create.jamThreshold, "-30")

# try to create E2 with invalid jam threshold
netedit.leftClick(referencePosition, 450, 220)

# set valid jam threshold
netedit.changeDefaultValue(netedit.attrs.E2.create.jamThreshold, "15.5")

# create E2 with valid jam threshold
netedit.leftClick(referencePosition, 450, 220)

# Check undo redo
netedit.undo(referencePosition, 1)
netedit.redo(referencePosition, 1)

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
