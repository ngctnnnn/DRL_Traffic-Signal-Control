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

# go to additional mode
netedit.additionalMode()

# select routeProbe
netedit.changeElement("routeProbe")

# create routeProbe
netedit.leftClick(referencePosition, 380, 215)

# recompute
netedit.rebuildNetwork()

# Change to delete
netedit.deleteMode()

# disable 'Automatically delete additionals'
netedit.changeProtectAdditionalElements(referencePosition)

# delete loaded routeProbe
netedit.leftClick(referencePosition, 326, 205)

# delete created routeProbe (using stack)
netedit.leftClick(referencePosition, 326, 205)

# delete lane with the second loaded routeProbe
netedit.leftClick(referencePosition, 280, 265)

# Check undo
netedit.undo(referencePosition, 3)

# Change to delete
netedit.deleteMode()

# enable 'Automatically delete additionals'
netedit.changeProtectAdditionalElements(referencePosition)

# try to delete lane with the second loaded routeProbe (doesn't allowed)
netedit.leftClick(referencePosition, 280, 265)

# wait warning
netedit.waitDeleteWarning()

# check redo
netedit.redo(referencePosition, 3)

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
