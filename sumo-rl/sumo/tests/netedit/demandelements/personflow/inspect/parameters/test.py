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
# @date    2019-07-16

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

# force save additionals
netedit.forceSaveAdditionals()

# go to demand mode
netedit.supermodeDemand()

# go to person mode
netedit.personMode()

# change Person
netedit.changeElement("personFlow")

# create person using three edges
netedit.leftClick(referencePosition, 274, 400)
netedit.leftClick(referencePosition, 180, 60)

# press enter to create person
netedit.typeEnter()

# change zoom
netedit.setZoom("0", "-6", "5")

# go to inspect mode
netedit.inspectMode()

# inspect person
netedit.leftClick(referencePosition, 310, 140)

# check parameters
netedit.checkParameters(referencePosition, netedit.attrs.personFlow.inspect.parameters, False)

# save network
netedit.saveNetwork(referencePosition)

# save persons
netedit.saveRoutes(referencePosition)

# save additionals
netedit.saveAdditionals(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
