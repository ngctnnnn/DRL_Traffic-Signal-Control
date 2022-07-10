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
neteditProcess, referencePosition = netedit.setupAndStart(neteditTestRoot, [], False)

# Rebuild network
netedit.rebuildNetwork()

# go to inspect mode
netedit.inspectMode()

# select first left edge and change their junction
netedit.leftClick(referencePosition, 250, 205)
netedit.modifyAttribute(netedit.attrs.edge.inspect.fromEdge, "B", False)
netedit.rebuildNetwork()

# select second left edge and change their junction
netedit.leftClick(referencePosition, 250, 255)
netedit.modifyAttribute(netedit.attrs.edge.inspect.toEdge, "A", False)
netedit.rebuildNetwork()

# select first right edge and change their junction
netedit.leftClick(referencePosition, 500, 205)
netedit.modifyAttribute(netedit.attrs.edge.inspect.toEdge, "B", False)
netedit.rebuildNetwork()

# select second right edge and change their junction
netedit.leftClick(referencePosition, 500, 255)
netedit.modifyAttribute(netedit.attrs.edge.inspect.fromEdge, "A", False)
netedit.rebuildNetwork()

# Check undo redo
netedit.undo(referencePosition, 4)
netedit.rebuildNetwork()

# Check redo
netedit.redo(referencePosition, 4)
netedit.rebuildNetwork()

# save additionals
netedit.saveAdditionals(referencePosition)

# save network
netedit.saveNetwork(referencePosition)

# quit netedit
netedit.quit(neteditProcess)
