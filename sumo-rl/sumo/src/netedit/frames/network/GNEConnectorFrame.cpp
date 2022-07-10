/****************************************************************************/
// Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
// Copyright (C) 2011-2022 German Aerospace Center (DLR) and others.
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License 2.0 which is available at
// https://www.eclipse.org/legal/epl-2.0/
// This Source Code may also be made available under the following Secondary
// Licenses when the conditions for such availability set forth in the Eclipse
// Public License 2.0 are satisfied: GNU General Public License, version 2
// or later which is available at
// https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
/****************************************************************************/
/// @file    GNEConnectorFrame.cpp
/// @author  Jakob Erdmann
/// @date    May 2011
///
// The Widget for modifying lane-to-lane connections
/****************************************************************************/
#include <config.h>

#include <utils/gui/windows/GUIAppEnum.h>
#include <utils/gui/div/GUIDesigns.h>
#include <netedit/changes/GNEChange_Connection.h>
#include <netedit/GNEViewParent.h>
#include <netedit/GNEUndoList.h>
#include <netedit/GNENet.h>
#include <netedit/GNEViewNet.h>
#include <netedit/elements/network/GNEConnection.h>
#include <netedit/frames/network/GNEConnectorFrame.h>
#include <netedit/frames/common/GNESelectorFrame.h>


// ===========================================================================
// FOX callback mapping
// ===========================================================================

FXDEFMAP(GNEConnectorFrame::ConnectionModifications) ConnectionModificationsMap[] = {
    FXMAPFUNC(SEL_COMMAND,  MID_CANCEL,     GNEConnectorFrame::ConnectionModifications::onCmdCancelModifications),
    FXMAPFUNC(SEL_COMMAND,  MID_OK,         GNEConnectorFrame::ConnectionModifications::onCmdSaveModifications),
};

FXDEFMAP(GNEConnectorFrame::ConnectionOperations) ConnectionOperationsMap[] = {
    FXMAPFUNC(SEL_COMMAND,  MID_CHOOSEN_CLEAR,                          GNEConnectorFrame::ConnectionOperations::onCmdClearSelectedConnections),
    FXMAPFUNC(SEL_COMMAND,  MID_CHOOSEN_RESET,                          GNEConnectorFrame::ConnectionOperations::onCmdResetSelectedConnections),
    FXMAPFUNC(SEL_COMMAND,  MID_GNE_CONNECTORFRAME_SELECTDEADENDS,      GNEConnectorFrame::ConnectionOperations::onCmdSelectDeadEnds),
    FXMAPFUNC(SEL_COMMAND,  MID_GNE_CONNECTORFRAME_SELECTDEADSTARTS,    GNEConnectorFrame::ConnectionOperations::onCmdSelectDeadStarts),
    FXMAPFUNC(SEL_COMMAND,  MID_GNE_CONNECTORFRAME_SELECTCONFLICTS,     GNEConnectorFrame::ConnectionOperations::onCmdSelectConflicts),
    FXMAPFUNC(SEL_COMMAND,  MID_GNE_CONNECTORFRAME_SELECTPASS,          GNEConnectorFrame::ConnectionOperations::onCmdSelectPass),
};

// Object implementation
FXIMPLEMENT(GNEConnectorFrame::ConnectionModifications, FXGroupBoxModule, ConnectionModificationsMap, ARRAYNUMBER(ConnectionModificationsMap))
FXIMPLEMENT(GNEConnectorFrame::ConnectionOperations,    FXGroupBoxModule, ConnectionOperationsMap,    ARRAYNUMBER(ConnectionOperationsMap))


// ===========================================================================
// method definitions
// ===========================================================================

// ---------------------------------------------------------------------------
// GNEConnectorFrame::CurrentLane - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::CurrentLane::CurrentLane(GNEConnectorFrame* connectorFrameParent) :
    FXGroupBoxModule(connectorFrameParent, "Lane") {
    // create lane label
    myCurrentLaneLabel = new FXLabel(getCollapsableFrame(), "No lane selected", 0, GUIDesignLabelLeft);
}


GNEConnectorFrame::CurrentLane::~CurrentLane() {}


void
GNEConnectorFrame::CurrentLane::updateCurrentLaneLabel(const std::string& laneID) {
    if (laneID.empty()) {
        myCurrentLaneLabel->setText("No lane selected");
    } else {
        myCurrentLaneLabel->setText((std::string("Current Lane: ") + laneID).c_str());
    }
}

// ---------------------------------------------------------------------------
// GNEConnectorFrame::ConnectionModifications - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::ConnectionModifications::ConnectionModifications(GNEConnectorFrame* connectorFrameParent) :
    FXGroupBoxModule(connectorFrameParent, "Modifications"),
    myConnectorFrameParent(connectorFrameParent) {

    // Create "Cancel" button
    myCancelButton = new FXButton(getCollapsableFrame(), "Cancel\t\tDiscard connection modifications (Esc)",
                                  GUIIconSubSys::getIcon(GUIIcon::CANCEL), this, MID_CANCEL, GUIDesignButton);
    // Create "OK" button
    mySaveButton = new FXButton(getCollapsableFrame(), "OK\t\tSave connection modifications (Enter)",
                                GUIIconSubSys::getIcon(GUIIcon::ACCEPT), this, MID_OK, GUIDesignButton);

    // Create checkbox for protect routes
    myProtectRoutesCheckBox = new FXCheckButton(getCollapsableFrame(), "Protect routes", this, MID_GNE_SET_ATTRIBUTE, GUIDesignCheckButton);
}


GNEConnectorFrame::ConnectionModifications::~ConnectionModifications() {}


long
GNEConnectorFrame::ConnectionModifications::onCmdCancelModifications(FXObject*, FXSelector, void*) {
    if (myConnectorFrameParent->myCurrentEditedLane != 0) {
        myConnectorFrameParent->getViewNet()->getUndoList()->abortAllChangeGroups();
        if (myConnectorFrameParent->myNumChanges) {
            myConnectorFrameParent->getViewNet()->setStatusBarText("Changes reverted");
        }
        myConnectorFrameParent->cleanup();
        myConnectorFrameParent->getViewNet()->updateViewNet();
    }
    return 1;
}


long
GNEConnectorFrame::ConnectionModifications::onCmdSaveModifications(FXObject*, FXSelector, void*) {
    if (myConnectorFrameParent->myCurrentEditedLane != 0) {
        // check if routes has to be protected
        if (myProtectRoutesCheckBox->isEnabled() && (myProtectRoutesCheckBox->getCheck() == TRUE)) {
            for (const auto& i : myConnectorFrameParent->myCurrentEditedLane->getParentEdge()->getChildDemandElements()) {
                if (i->isDemandElementValid() != GNEDemandElement::Problem::OK) {
                    FXMessageBox::warning(getApp(), MBOX_OK,
                                          "Error saving connection operations", "%s",
                                          ("Connection edition  cannot be saved because route '" + i->getID() + "' is broken.").c_str());
                    return 1;
                }
            }
        }
        // finish route editing
        myConnectorFrameParent->getViewNet()->getUndoList()->end();
        if (myConnectorFrameParent->myNumChanges) {
            myConnectorFrameParent->getViewNet()->setStatusBarText("Changes accepted");
        }
        myConnectorFrameParent->cleanup();
        myConnectorFrameParent->getViewNet()->updateViewNet();
    }
    return 1;
}

// ---------------------------------------------------------------------------
// GNEConnectorFrame::ConnectionOperations - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::ConnectionOperations::ConnectionOperations(GNEConnectorFrame* connectorFrameParent) :
    FXGroupBoxModule(connectorFrameParent, "Operations"),
    myConnectorFrameParent(connectorFrameParent) {

    // Create "Select Dead Ends" button
    mySelectDeadEndsButton = new FXButton(getCollapsableFrame(), "Select Dead Ends\t\tSelects all lanes that have no outgoing connection (clears previous selection)",
                                          0, this, MID_GNE_CONNECTORFRAME_SELECTDEADENDS, GUIDesignButton);
    // Create "Select Dead Starts" button
    mySelectDeadStartsButton = new FXButton(getCollapsableFrame(), "Select Dead Starts\t\tSelects all lanes that have no incoming connection (clears previous selection)",
                                            0, this, MID_GNE_CONNECTORFRAME_SELECTDEADSTARTS, GUIDesignButton);
    // Create "Select Conflicts" button
    mySelectConflictsButton = new FXButton(getCollapsableFrame(), "Select Conflicts\t\tSelects all lanes with more than one incoming connection from the same edge (clears previous selection)",
                                           0, this, MID_GNE_CONNECTORFRAME_SELECTCONFLICTS, GUIDesignButton);
    // Create "Select Edges which may always pass" button
    mySelectPassingButton = new FXButton(getCollapsableFrame(), "Select Passing\t\tSelects all lanes with a connection that has has the 'pass' attribute set",
                                         0, this, MID_GNE_CONNECTORFRAME_SELECTPASS, GUIDesignButton);
    // Create "Clear Selected" button
    myClearSelectedButton = new FXButton(getCollapsableFrame(), "Clear Selected\t\tClears all connections of all selected objects",
                                         0, this, MID_CHOOSEN_CLEAR, GUIDesignButton);
    // Create "Reset Selected" button
    myResetSelectedButton = new FXButton(getCollapsableFrame(), "Reset Selected\t\tRecomputes connections at all selected junctions",
                                         0, this, MID_CHOOSEN_RESET, GUIDesignButton);
}


GNEConnectorFrame::ConnectionOperations::~ConnectionOperations() {}


long
GNEConnectorFrame::ConnectionOperations::onCmdSelectDeadEnds(FXObject*, FXSelector, void*) {
    // select all lanes that have no successor lane
    std::vector<GNEAttributeCarrier*> deadEnds;
    // every edge knows its outgoing connections so we can look at each edge in isolation
    for (const auto& edge : myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getEdges()) {
        for (const auto& lane : edge.second->getLanes()) {
            if (edge.second->getNBEdge()->getConnectionsFromLane(lane->getIndex()).size() == 0) {
                deadEnds.push_back(lane);
            }
        }
    }
    myConnectorFrameParent->getViewNet()->getViewParent()->getSelectorFrame()->handleIDs(deadEnds, GNESelectorFrame::ModificationMode::Operation::REPLACE);
    return 1;
}


long
GNEConnectorFrame::ConnectionOperations::onCmdSelectDeadStarts(FXObject*, FXSelector, void*) {
    // select all lanes that have no predecessor lane
    std::set<GNEAttributeCarrier*> deadStarts;
    GNENet* net = myConnectorFrameParent->getViewNet()->getNet();
    // every edge knows only its outgoing connections so we look at whole junctions
    for (const auto& junction : myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getJunctions()) {
        // first collect all outgoing lanes
        for (const auto& outgoingEdge : junction.second->getGNEOutgoingEdges()) {
            for (const auto& lane : outgoingEdge->getLanes()) {
                deadStarts.insert(lane);
            }
        }
        // then remove all approached lanes
        for (const auto& incomingEdge : junction.second->getGNEIncomingEdges()) {
            for (const auto& connection : incomingEdge->getNBEdge()->getConnections()) {
                deadStarts.erase(net->getAttributeCarriers()->retrieveEdge(connection.toEdge->getID())->getLanes()[connection.toLane]);
            }
        }
    }
    std::vector<GNEAttributeCarrier*> selectObjects(deadStarts.begin(), deadStarts.end());
    myConnectorFrameParent->getViewNet()->getViewParent()->getSelectorFrame()->handleIDs(selectObjects, GNESelectorFrame::ModificationMode::Operation::REPLACE);
    return 1;
}


long
GNEConnectorFrame::ConnectionOperations::onCmdSelectConflicts(FXObject*, FXSelector, void*) {
    std::vector<GNEAttributeCarrier*> conflicts;
    // conflicts happen per edge so we can look at each edge in isolation
    for (const auto& edge : myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getEdges()) {
        const EdgeVector destinations = edge.second->getNBEdge()->getConnectedEdges();
        for (const auto& destination : destinations) {
            GNEEdge* dest = myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->retrieveEdge(destination->getID());
            for (const auto& lane : dest->getLanes()) {
                const bool isConflicted = count_if(edge.second->getNBEdge()->getConnections().begin(), edge.second->getNBEdge()->getConnections().end(),
                                                   NBEdge::connections_toedgelane_finder(destination, (int)lane->getIndex(), -1)) > 1;
                if (isConflicted) {
                    conflicts.push_back(lane);
                }
            }
        }

    }
    myConnectorFrameParent->getViewNet()->getViewParent()->getSelectorFrame()->handleIDs(conflicts, GNESelectorFrame::ModificationMode::Operation::REPLACE);
    return 1;
}


long
GNEConnectorFrame::ConnectionOperations::onCmdSelectPass(FXObject*, FXSelector, void*) {
    std::vector<GNEAttributeCarrier*> pass;
    for (const auto& edge : myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getEdges()) {
        for (const auto& connection : edge.second->getNBEdge()->getConnections()) {
            if (connection.mayDefinitelyPass) {
                pass.push_back(edge.second->getLanes()[connection.fromLane]);
            }
        }
    }
    myConnectorFrameParent->getViewNet()->getViewParent()->getSelectorFrame()->handleIDs(pass, GNESelectorFrame::ModificationMode::Operation::REPLACE);
    return 1;
}


long
GNEConnectorFrame::ConnectionOperations::onCmdClearSelectedConnections(FXObject*, FXSelector, void*) {
    myConnectorFrameParent->myConnectionModifications->onCmdCancelModifications(0, 0, 0);
    myConnectorFrameParent->getViewNet()->getUndoList()->begin(GUIIcon::CONNECTION, "clear connections from selected lanes, edges and " + toString(SUMO_TAG_JUNCTION) + "s");
    // clear junction's connection
    const auto selectedJunctions = myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getSelectedJunctions();
    for (const auto& junction : selectedJunctions) {
        junction->setLogicValid(false, myConnectorFrameParent->getViewNet()->getUndoList()); // clear connections
        junction->setLogicValid(false, myConnectorFrameParent->getViewNet()->getUndoList(), GNEAttributeCarrier::FEATURE_MODIFIED); // prevent re-guessing
    }
    // clear edge's connection
    const auto selectedEdges = myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getSelectedEdges();
    for (const auto& edge : selectedEdges) {
        for (const auto& lane : edge->getLanes()) {
            myConnectorFrameParent->removeConnections(lane);
        }
    }
    // clear lane's connection
    const auto selectedLanes = myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getSelectedLanes();
    for (const auto& lane : selectedLanes) {
        myConnectorFrameParent->removeConnections(lane);
    }
    myConnectorFrameParent->getViewNet()->getUndoList()->end();
    return 1;
}


long
GNEConnectorFrame::ConnectionOperations::onCmdResetSelectedConnections(FXObject*, FXSelector, void*) {
    myConnectorFrameParent->myConnectionModifications->onCmdCancelModifications(0, 0, 0);
    myConnectorFrameParent->getViewNet()->getUndoList()->begin(GUIIcon::CONNECTION, "reset connections from selected lanes");
    const auto selectedJunctions = myConnectorFrameParent->getViewNet()->getNet()->getAttributeCarriers()->getSelectedJunctions();
    for (const auto& junction : selectedJunctions) {
        junction->setLogicValid(false, myConnectorFrameParent->getViewNet()->getUndoList());
    }
    myConnectorFrameParent->getViewNet()->getUndoList()->end();
    if (selectedJunctions.size() > 0) {
        auto viewNet = myConnectorFrameParent->getViewNet();
        viewNet->getNet()->requireRecompute();
        viewNet->getNet()->computeNetwork(viewNet->getViewParent()->getGNEAppWindows());
    }
    return 1;
}

// ---------------------------------------------------------------------------
// GNEConnectorFrame::ConnectionSelection - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::ConnectionSelection::ConnectionSelection(GNEConnectorFrame* connectorFrameParent) :
    FXGroupBoxModule(connectorFrameParent, "Selection") {
    // create Selection Hint
    myHoldShiftLabel = new FXLabel(getCollapsableFrame(), "Hold <SHIFT> while clicking\nto create unyielding\nconnections (pass=true).", 0, GUIDesignLabelFrameInformation);
    myHoldControlLabel = new FXLabel(getCollapsableFrame(), "Hold <CTRL> while clicking\nto create conflicting\nconnections (i.e. at zipper\nnodes or with incompatible\npermissions)", 0, GUIDesignLabelFrameInformation);
}


GNEConnectorFrame::ConnectionSelection::~ConnectionSelection() {}

// ---------------------------------------------------------------------------
// GNEConnectorFrame::ConnectionLegend - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::Legend::Legend(GNEConnectorFrame* connectorFrameParent) :
    FXGroupBoxModule(connectorFrameParent, "Information") {

    // create possible target label
    FXLabel* possibleTargetLabel = new FXLabel(getCollapsableFrame(), "Possible Target", 0, GUIDesignLabelLeft);
    possibleTargetLabel->setBackColor(MFXUtils::getFXColor(connectorFrameParent->getViewNet()->getVisualisationSettings().candidateColorSettings.possible));
    possibleTargetLabel->setTextColor(MFXUtils::getFXColor(RGBColor::WHITE));

    // create source label
    FXLabel* sourceLabel = new FXLabel(getCollapsableFrame(), "Source lane", 0, GUIDesignLabelLeft);
    sourceLabel->setBackColor(MFXUtils::getFXColor(connectorFrameParent->getViewNet()->getVisualisationSettings().candidateColorSettings.source));

    // create target label
    FXLabel* targetLabel = new FXLabel(getCollapsableFrame(), "Target lane", 0, GUIDesignLabelLeft);
    targetLabel->setBackColor(MFXUtils::getFXColor(connectorFrameParent->getViewNet()->getVisualisationSettings().candidateColorSettings.target));

    // create target (pass) label
    FXLabel* targetPassLabel = new FXLabel(getCollapsableFrame(), "Target (pass)", 0, GUIDesignLabelLeft);
    targetPassLabel->setBackColor(MFXUtils::getFXColor(connectorFrameParent->getViewNet()->getVisualisationSettings().candidateColorSettings.special));

    // create conflict label
    FXLabel* conflictLabel = new FXLabel(getCollapsableFrame(), "Conflict", 0, GUIDesignLabelLeft);
    conflictLabel->setBackColor(MFXUtils::getFXColor(connectorFrameParent->getViewNet()->getVisualisationSettings().candidateColorSettings.conflict));
}


GNEConnectorFrame::Legend::~Legend() {}

// ---------------------------------------------------------------------------
// GNEConnectorFrame - methods
// ---------------------------------------------------------------------------

GNEConnectorFrame::GNEConnectorFrame(FXHorizontalFrame* horizontalFrameParent, GNEViewNet* viewNet):
    GNEFrame(horizontalFrameParent, viewNet, "Edit Connections"),
    myCurrentEditedLane(0),
    myNumChanges(0) {
    // create current lane modul
    myCurrentLane = new CurrentLane(this);

    // create connection modifications modul
    myConnectionModifications = new ConnectionModifications(this);

    // create connection operations modul
    myConnectionOperations = new ConnectionOperations(this);

    // create connection selection modul
    myConnectionSelection = new ConnectionSelection(this);

    // create connection legend modul
    myLegend = new Legend(this);
}


GNEConnectorFrame::~GNEConnectorFrame() {}


void
GNEConnectorFrame::handleLaneClick(const GNEViewNetHelper::ObjectsUnderCursor& objectsUnderCursor) {
    // get lane front
    GNELane* clickedLane = objectsUnderCursor.getLaneFront();
    // iterate over lanes
    for (const auto& lane : objectsUnderCursor.getLanes()) {
        // if parent edge of lane is front element, update clickedLane
        if (lane->getParentEdge() == myViewNet->getFrontAttributeCarrier()) {
            clickedLane = lane;
        }
    }
    // build connection
    buildConnection(clickedLane, myViewNet->getMouseButtonKeyPressed().shiftKeyPressed(), myViewNet->getMouseButtonKeyPressed().controlKeyPressed(), true);
}


GNEConnectorFrame::ConnectionModifications*
GNEConnectorFrame::getConnectionModifications() const {
    return myConnectionModifications;
}


void
GNEConnectorFrame::removeConnections(GNELane* lane) {
    // select lane as current lane
    buildConnection(lane, false, false, true); // select as current lane
    // iterate over all potential targets
    for (const auto& potentialTarget : myPotentialTargets) {
        // remove connections using the apropiate parameters in function "buildConnection"
        buildConnection(potentialTarget, false, false, false);
    }
    // save modifications
    myConnectionModifications->onCmdSaveModifications(0, 0, 0);
}


void
GNEConnectorFrame::buildConnection(GNELane* lane, const bool mayDefinitelyPass, const bool allowConflict, const bool toggle) {
    if (myCurrentEditedLane == 0) {
        myCurrentEditedLane = lane;
        myCurrentEditedLane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.source);
        initTargets();
        myNumChanges = 0;
        myViewNet->getUndoList()->begin(GUIIcon::CONNECTION, "modify " + toString(SUMO_TAG_CONNECTION) + "s");
    } else if (myPotentialTargets.count(lane)
               || (allowConflict && lane->getParentEdge()->getFromJunction() == myCurrentEditedLane->getParentEdge()->getToJunction())) {
        const int fromIndex = myCurrentEditedLane->getIndex();
        GNEEdge* srcEdge = myCurrentEditedLane->getParentEdge();
        GNEEdge* destEdge = lane->getParentEdge();
        std::vector<NBEdge::Connection> connections = srcEdge->getNBEdge()->getConnectionsFromLane(fromIndex);
        bool changed = false;
        // get lane status
        LaneStatus status = getLaneStatus(connections, lane);
        if (status == LaneStatus::CONFLICTED && allowConflict) {
            status = LaneStatus::UNCONNECTED;
        }
        // create depending of status
        switch (status) {
            case LaneStatus::UNCONNECTED:
                if (toggle) {
                    // create new connection
                    NBEdge::Connection newCon(fromIndex, destEdge->getNBEdge(), lane->getIndex(), mayDefinitelyPass);
                    // if the connection was previously deleted (by clicking the same lane twice), restore all values
                    for (NBEdge::Connection& c : myDeletedConnections) {
                        // fromLane must be the same, only check toLane
                        if (c.toEdge == destEdge->getNBEdge() && c.toLane == lane->getIndex()) {
                            newCon = c;
                            newCon.mayDefinitelyPass = mayDefinitelyPass;
                        }
                    }
                    NBConnection newNBCon(srcEdge->getNBEdge(), fromIndex, destEdge->getNBEdge(), lane->getIndex(), newCon.tlLinkIndex);
                    myViewNet->getUndoList()->add(new GNEChange_Connection(srcEdge, newCon, false, true), true);
                    if (mayDefinitelyPass) {
                        lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.special);
                    } else {
                        lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.target);
                    }
                    srcEdge->getToJunction()->invalidateTLS(myViewNet->getUndoList(), NBConnection::InvalidConnection, newNBCon);
                }
                break;
            case LaneStatus::CONNECTED:
            case LaneStatus::CONNECTED_PASS: {
                // remove connection
                GNEConnection* con = srcEdge->retrieveGNEConnection(fromIndex, destEdge->getNBEdge(), lane->getIndex());
                myDeletedConnections.push_back(con->getNBEdgeConnection());
                myViewNet->getNet()->deleteConnection(con, myViewNet->getUndoList());
                lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.possible);
                changed = true;
                break;
            }
            case LaneStatus::CONFLICTED:
                SVCPermissions fromPermissions = srcEdge->getNBEdge()->getPermissions(fromIndex);
                SVCPermissions toPermissions = destEdge->getNBEdge()->getPermissions(lane->getIndex());
                if ((fromPermissions & toPermissions) == SVC_PEDESTRIAN) {
                    myViewNet->setStatusBarText("Pedestrian connections are generated automatically");
                } else if ((fromPermissions & toPermissions) == 0) {
                    myViewNet->setStatusBarText("Incompatible vehicle class permissions");
                } else {
                    myViewNet->setStatusBarText("Another lane from the same edge already connects to that lane");
                }
                break;
        }
        if (changed) {
            myNumChanges += 1;
        }
    } else {
        myViewNet->setStatusBarText("Invalid target for " + toString(SUMO_TAG_CONNECTION));
    }
    myCurrentLane->updateCurrentLaneLabel(myCurrentEditedLane->getID());
}


void
GNEConnectorFrame::initTargets() {
    // gather potential targets
    NBNode* nbn = myCurrentEditedLane->getParentEdge()->getToJunction()->getNBNode();
    // get potencial targets
    for (const auto& NBEEdge : nbn->getOutgoingEdges()) {
        GNEEdge* edge = myViewNet->getNet()->getAttributeCarriers()->retrieveEdge(NBEEdge->getID());
        for (const auto& lane : edge->getLanes()) {
            myPotentialTargets.insert(lane);
        }
    }
    // set color for existing connections
    std::vector<NBEdge::Connection> connections = myCurrentEditedLane->getParentEdge()->getNBEdge()->getConnectionsFromLane(myCurrentEditedLane->getIndex());
    for (const auto& lane : myPotentialTargets) {
        switch (getLaneStatus(connections, lane)) {
            case LaneStatus::CONNECTED:
                lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.target);
                break;
            case LaneStatus::CONNECTED_PASS:
                lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.special);
                break;
            case LaneStatus::CONFLICTED:
                lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.conflict);
                break;
            case LaneStatus::UNCONNECTED:
                lane->setSpecialColor(&myViewNet->getVisualisationSettings().candidateColorSettings.possible);
                break;
        }
    }
}


void
GNEConnectorFrame::cleanup() {
    // restore colors of potential targets
    for (auto it : myPotentialTargets) {
        it->setSpecialColor(0);
    }
    // clear attributes
    myPotentialTargets.clear();
    myNumChanges = 0;
    myCurrentEditedLane->setSpecialColor(0);
    myCurrentEditedLane = nullptr;
    myDeletedConnections.clear();
    myCurrentLane->updateCurrentLaneLabel("");
}


GNEConnectorFrame::LaneStatus
GNEConnectorFrame::getLaneStatus(const std::vector<NBEdge::Connection>& connections, const GNELane* targetLane) const {
    NBEdge* srcEdge = myCurrentEditedLane->getParentEdge()->getNBEdge();
    const int fromIndex = myCurrentEditedLane->getIndex();
    NBEdge* destEdge = targetLane->getParentEdge()->getNBEdge();
    const int toIndex = targetLane->getIndex();
    std::vector<NBEdge::Connection>::const_iterator con_it = find_if(
                connections.begin(), connections.end(),
                NBEdge::connections_finder(fromIndex, destEdge, toIndex));
    const bool isConnected = con_it != connections.end();
    if (isConnected) {
        if (con_it->mayDefinitelyPass) {
            return LaneStatus::CONNECTED_PASS;
        } else {
            return LaneStatus::CONNECTED;
        }
    } else if (srcEdge->hasConnectionTo(destEdge, toIndex)
               || (srcEdge->getPermissions(fromIndex) & destEdge->getPermissions(toIndex) & ~SVC_PEDESTRIAN) == 0) {
        return LaneStatus::CONFLICTED;
    } else {
        return LaneStatus::UNCONNECTED;
    }
}


/****************************************************************************/
