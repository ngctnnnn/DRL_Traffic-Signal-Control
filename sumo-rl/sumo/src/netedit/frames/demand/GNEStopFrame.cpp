/****************************************************************************/
// Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
// Copyright (C) 2001-2022 German Aerospace Center (DLR) and others.
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
/// @file    GNEStopFrame.cpp
/// @author  Pablo Alvarez Lopez
/// @date    March 2019
///
// The Widget for add Stops elements
/****************************************************************************/
#include <config.h>

#include <netedit/GNENet.h>
#include <netedit/GNEViewNet.h>
#include <utils/gui/div/GUIDesigns.h>

#include "GNEStopFrame.h"

// ===========================================================================
// method definitions
// ===========================================================================

// ---------------------------------------------------------------------------
// GNEStopFrame::HelpCreation - methods
// ---------------------------------------------------------------------------

GNEStopFrame::HelpCreation::HelpCreation(GNEStopFrame* StopFrameParent) :
    FXGroupBoxModule(StopFrameParent, "Help"),
    myStopFrameParent(StopFrameParent) {
    myInformationLabel = new FXLabel(getCollapsableFrame(), "", 0, GUIDesignLabelFrameInformation);
}


GNEStopFrame::HelpCreation::~HelpCreation() {}


void
GNEStopFrame::HelpCreation::showHelpCreation() {
    // first update help creation
    updateHelpCreation();
    // show module
    show();
}


void
GNEStopFrame::HelpCreation::hideHelpCreation() {
    hide();
}


void
GNEStopFrame::HelpCreation::updateHelpCreation() {
    // create information label
    std::ostringstream information;
    // set text depending of selected Stop type
    if (myStopFrameParent->myStopTagSelector->getCurrentTemplateAC()) {
        switch (myStopFrameParent->myStopTagSelector->getCurrentTemplateAC()->getTagProperty().getTag()) {
            case SUMO_TAG_STOP_BUSSTOP:
                information
                        << "- Click over a bus stop\n"
                        << "  to create a stop.";
                break;
            case SUMO_TAG_STOP_CONTAINERSTOP:
                information
                        << "- Click over a container stop\n"
                        << "  to create a stop.";
                break;
            case SUMO_TAG_STOP_CHARGINGSTATION:
                information
                        << "- Click over a charging \n"
                        << "  station to create a stop.";
                break;
            case SUMO_TAG_STOP_PARKINGAREA:
                information
                        << "- Click over a parking area\n"
                        << "  to create a stop.";
                break;
            case SUMO_TAG_STOP_LANE:
                information
                        << "- Click over a lane to\n"
                        << "  create a stop.";
                break;
            default:
                information
                        << "- No stop parents in\n"
                        << "  current network.";
                break;
        }
    }
    // set information label
    myInformationLabel->setText(information.str().c_str());
}

// ---------------------------------------------------------------------------
// GNEStopFrame - methods
// ---------------------------------------------------------------------------

GNEStopFrame::GNEStopFrame(FXHorizontalFrame* horizontalFrameParent, GNEViewNet* viewNet) :
    GNEFrame(horizontalFrameParent, viewNet, "Stops"),
    myRouteHandler("", viewNet->getNet(), true),
    myStopParentBaseObject(new CommonXMLStructure::SumoBaseObject(nullptr)) {

    // Create Stop parent selector
    myStopParentSelector = new DemandElementSelector(this, {GNETagProperties::TagType::VEHICLE, GNETagProperties::TagType::ROUTE});

    // Create item Selector module for Stops
    myStopTagSelector = new GNETagSelector(this, GNETagProperties::TagType::STOP, SUMO_TAG_STOP_LANE);

    // Create Stop parameters
    myStopAttributes = new GNEAttributesCreator(this);

    // Create Netedit parameter
    myNeteditAttributes = new GNENeteditAttributes(this);

    // Create Help Creation Module
    myHelpCreation = new HelpCreation(this);

    // refresh myStopParentMatchBox
    myStopParentSelector->refreshDemandElementSelector();
}


GNEStopFrame::~GNEStopFrame() {
    delete myStopParentBaseObject;
}


void
GNEStopFrame::show() {
    // first check if stop frame modules can be shown
    bool validStopParent = false;
    // check if at least there an item that supports an stop
    for (auto i = myStopParentSelector->getAllowedTags().begin(); (i != myStopParentSelector->getAllowedTags().end()) && (validStopParent == false); i++) {
        if (myViewNet->getNet()->getAttributeCarriers()->getDemandElements().at(*i).size() > 0) {
            validStopParent = true;
        }
    }
    // show or hide modules depending of validStopParent
    if (validStopParent) {
        // refresh tag selector
        myStopTagSelector->refreshTagSelector();
        // refresh vType selector
        myStopParentSelector->refreshDemandElementSelector();
        // refresh tag selector
        myStopTagSelector->refreshTagSelector();
        // show
        myStopParentSelector->showDemandElementSelector();
        myStopTagSelector->showTagSelector();
    } else {
        // hide modules (except help creation)
        myStopParentSelector->hideDemandElementSelector();
        myStopTagSelector->hideTagSelector();
        myStopAttributes->hideAttributesCreatorModule();
        myNeteditAttributes->hideNeteditAttributesModule();
        // show help creation module
        myHelpCreation->showHelpCreation();
    }
    // show frame
    GNEFrame::show();
}


bool
GNEStopFrame::addStop(const GNEViewNetHelper::ObjectsUnderCursor& objectsUnderCursor, const GNEViewNetHelper::MouseButtonKeyPressed& mouseButtonKeyPressed) {
    // first check stop type
    if (myStopTagSelector->getCurrentTemplateAC() == nullptr) {
        WRITE_WARNING("Selected Stop type isn't valid.");
        return false;
    }
    // check if we're selecting a new stop parent
    if (mouseButtonKeyPressed.shiftKeyPressed()) {
        if (objectsUnderCursor.getDemandElementFront() &&
                (objectsUnderCursor.getDemandElementFront()->getTagProperty().isVehicle() || objectsUnderCursor.getDemandElementFront()->getTagProperty().getTag() == SUMO_TAG_ROUTE)) {
            myStopParentSelector->setDemandElement(objectsUnderCursor.getDemandElementFront());
            WRITE_WARNING("Selected " + objectsUnderCursor.getDemandElementFront()->getTagStr() + " '" + objectsUnderCursor.getDemandElementFront()->getID() + "' as stop parent.");
            return true;
        } else {
            WRITE_WARNING("Selected Stop parent isn't valid.");
            return false;
        }
    } else {
        // now check if stop parent selector is valid
        if (myStopParentSelector->getCurrentDemandElement() == nullptr) {
            WRITE_WARNING("Current selected Stop parent isn't valid.");
            return false;
        }
        // create stop base object
        getStopParameter(myStopTagSelector->getCurrentTemplateAC()->getTagProperty().getTag(),
                         objectsUnderCursor.getLaneFront(), objectsUnderCursor.getAdditionalFront());
        if (myStopParentBaseObject->getTag() != SUMO_TAG_NOTHING) {
            myRouteHandler.buildStop(myStopParentBaseObject->getSumoBaseObjectChildren().front(),
                                     myStopParentBaseObject->getSumoBaseObjectChildren().front()->getStopParameter());
            // show all trips
            if (myStopTagSelector->getCurrentTemplateAC()->getTagProperty().isStop()) {
                myViewNet->getDemandViewOptions().menuCheckShowAllTrips->setChecked(TRUE);
            } else {
                myViewNet->getDemandViewOptions().menuCheckShowAllPersonPlans->setChecked(TRUE);
            }
            // stop successfully created, then return true
            return true;
        } else {
            return false;
        }
    }
}

bool
GNEStopFrame::getStopParameter(const SumoXMLTag stopTag, const GNELane* lane, const GNEAdditional* stoppingPlace) {
    // first clear stop base object
    myStopParentBaseObject->clear();
    // declare stop parameters
    SUMOVehicleParameter::Stop stop;
    // first check that current selected Stop is valid
    if (stopTag == SUMO_TAG_NOTHING) {
        WRITE_WARNING("Current selected Stop type isn't valid.");
        return false;
    } else if ((stopTag == SUMO_TAG_STOP_LANE) || (stopTag == GNE_TAG_WAYPOINT_LANE)) {
        if (lane) {
            stop.lane = lane->getID();
            if ((stopTag == GNE_TAG_WAYPOINT_LANE) && (stop.speed == 0)) {
                stop.speed = lane->getSpeed();
            }
        } else {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_LANE) + " to create a stop placed in a " + toString(SUMO_TAG_LANE));
            return false;
        }
    } else if (stopTag == GNE_TAG_STOPPERSON_EDGE) {
        if (lane) {
            stop.edge = lane->getParentEdge()->getID();
        } else {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_EDGE) + " to create a stop placed in a " + toString(SUMO_TAG_EDGE));
            return false;
        }
    } else if (stoppingPlace) {
        if (stoppingPlace->getTagProperty().getTag() == SUMO_TAG_BUS_STOP) {
            if ((stopTag != SUMO_TAG_STOP_BUSSTOP) && (stopTag != GNE_TAG_WAYPOINT_BUSSTOP) && (stopTag != GNE_TAG_STOPPERSON_BUSSTOP)) {
                WRITE_WARNING("Invalid clicked stopping place to create a stop placed in a " + stoppingPlace->getTagProperty().getTagStr());
                return false;
            } else {
                stop.busstop = stoppingPlace->getID();
                if ((stopTag == GNE_TAG_WAYPOINT_BUSSTOP) && (stop.speed == 0)) {
                    stop.speed = stoppingPlace->getParentLanes().front()->getSpeed();
                }
                stop.startPos = 0;
                stop.endPos = 0;
            }
        } else if (stoppingPlace->getTagProperty().getTag() == SUMO_TAG_CONTAINER_STOP) {
            if ((stopTag != SUMO_TAG_STOP_CONTAINERSTOP) && (stopTag != GNE_TAG_WAYPOINT_CONTAINERSTOP)) {
                WRITE_WARNING("Invalid clicked stopping place to create a stop placed in a " + stoppingPlace->getTagProperty().getTagStr());
                return false;
            } else {
                stop.containerstop = stoppingPlace->getID();
                if ((stopTag == GNE_TAG_WAYPOINT_CONTAINERSTOP) && (stop.speed == 0)) {
                    stop.speed = stoppingPlace->getParentLanes().front()->getSpeed();
                }
                stop.startPos = 0;
                stop.endPos = 0;
            }
        } else if (stoppingPlace->getTagProperty().getTag() == SUMO_TAG_CHARGING_STATION) {
            if ((stopTag != SUMO_TAG_STOP_CHARGINGSTATION) && (stopTag != GNE_TAG_WAYPOINT_CHARGINGSTATION)) {
                WRITE_WARNING("Invalid clicked stopping place to create a stop placed in a " + stoppingPlace->getTagProperty().getTagStr());
                return false;
            } else {
                stop.chargingStation = stoppingPlace->getID();
                if ((stopTag == GNE_TAG_WAYPOINT_CHARGINGSTATION) && (stop.speed == 0)) {
                    stop.speed = stoppingPlace->getParentLanes().front()->getSpeed();
                }
                stop.startPos = 0;
                stop.endPos = 0;
            }
        } else if (stoppingPlace->getTagProperty().getTag() == SUMO_TAG_PARKING_AREA) {
            if ((stopTag != SUMO_TAG_STOP_PARKINGAREA) && (stopTag != GNE_TAG_WAYPOINT_PARKINGAREA)) {
                WRITE_WARNING("Invalid clicked stopping place to create a stop placed in a " + stoppingPlace->getTagProperty().getTagStr());
                return false;
            } else {
                stop.parkingarea = stoppingPlace->getID();
                if ((stopTag == GNE_TAG_WAYPOINT_PARKINGAREA) && (stop.speed == 0)) {
                    stop.speed = stoppingPlace->getParentLanes().front()->getSpeed();
                }
                stop.startPos = 0;
                stop.endPos = 0;
            }
        }
    } else {
        if ((stopTag == SUMO_TAG_STOP_BUSSTOP) || (stopTag == GNE_TAG_WAYPOINT_BUSSTOP)) {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_STOP_BUSSTOP) + " to create a stop placed in a " + toString(SUMO_TAG_STOP_BUSSTOP));
        } else if ((stopTag == SUMO_TAG_STOP_CONTAINERSTOP) || (stopTag == GNE_TAG_WAYPOINT_CONTAINERSTOP)) {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_CONTAINER_STOP) + " to create a stop placed in a " + toString(SUMO_TAG_CONTAINER_STOP));
        } else if ((stopTag == SUMO_TAG_STOP_CHARGINGSTATION) || (stopTag == GNE_TAG_WAYPOINT_CHARGINGSTATION)) {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_CHARGING_STATION) + " to create a stop placed in a " + toString(SUMO_TAG_CHARGING_STATION));
        } else if ((stopTag == SUMO_TAG_STOP_PARKINGAREA) || (stopTag == GNE_TAG_WAYPOINT_PARKINGAREA)) {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_PARKING_AREA) + " to create a stop placed in a " + toString(SUMO_TAG_PARKING_AREA));
        } else if (stopTag == GNE_TAG_STOPPERSON_BUSSTOP) {
            WRITE_WARNING("Click over a " + toString(SUMO_TAG_STOP_BUSSTOP) + " to create a person stop placed in a " + toString(SUMO_TAG_STOP_BUSSTOP));
        }
        return false;
    }
    // check if stop attributes are valid
    if (!myStopAttributes->areValuesValid()) {
        myStopAttributes->showWarningMessage();
        return false;
    }
    // get stop parent
    const GNEDemandElement* stopParent = myStopParentSelector->getCurrentDemandElement();
    // if stopParent is a route, check that stop is placed over a route's edge
    if (stopParent->isRoute() && lane) {
        bool found = false;
        for (const auto& edge : stopParent->getParentEdges()) {
            if (edge == lane->getParentEdge()) {
                found = true;
            }
        }
        if (!found) {
            WRITE_WARNING("Stop must be placed over a route's edge");
            return false;
        }
    }
    // same if stoParent is a vehicle/flow with embedded route
    if (stopParent->getChildDemandElements().size() > 0 && stopParent->getChildDemandElements().front()->getTagProperty().isRoute() && lane) {
        bool found = false;
        for (const auto& edge : stopParent->getChildDemandElements().front()->getParentEdges()) {
            if (edge == lane->getParentEdge()) {
                found = true;
            }
        }
        if (!found) {
            WRITE_WARNING("Stop must be placed over an embeded route's edge");
            return false;
        }
    }
    // set parent tag and id
    myStopParentBaseObject->setTag(stopParent->getTagProperty().getTag());
    myStopParentBaseObject->addStringAttribute(SUMO_ATTR_ID, stopParent->getID());
    // add route, from and to attributes
    if (stopParent->getTagProperty().hasAttribute(SUMO_ATTR_ROUTE)) {
        myStopParentBaseObject->addStringAttribute(SUMO_ATTR_ROUTE, stopParent->getAttribute(SUMO_ATTR_ROUTE));
    }
    if (stopParent->getTagProperty().hasAttribute(SUMO_ATTR_FROM)) {
        myStopParentBaseObject->addStringAttribute(SUMO_ATTR_FROM, stopParent->getAttribute(SUMO_ATTR_FROM));
    }
    if (stopParent->getTagProperty().hasAttribute(SUMO_ATTR_TO)) {
        myStopParentBaseObject->addStringAttribute(SUMO_ATTR_TO, stopParent->getAttribute(SUMO_ATTR_TO));
    }
    // create stop object
    CommonXMLStructure::SumoBaseObject* stopBaseObject = new CommonXMLStructure::SumoBaseObject(myStopParentBaseObject);
    // get stop attributes
    myStopAttributes->getAttributesAndValues(stopBaseObject, true);
    // add netedit values
    if (!stop.lane.empty()) {
        myNeteditAttributes->getNeteditAttributesAndValues(stopBaseObject, lane);
        // check if start position can be parsed
        if (stopBaseObject->hasDoubleAttribute(SUMO_ATTR_STARTPOS)) {
            stop.startPos = stopBaseObject->getDoubleAttribute(SUMO_ATTR_STARTPOS);
            stop.parametersSet |= STOP_START_SET;
        }
        // check if end position can be parsed
        if (stopBaseObject->hasDoubleAttribute(SUMO_ATTR_ENDPOS)) {
            stop.endPos = stopBaseObject->getDoubleAttribute(SUMO_ATTR_ENDPOS);
            stop.parametersSet |= STOP_END_SET;
        }
    }
    // obtain friendly position
    if (stopBaseObject->hasBoolAttribute(SUMO_ATTR_FRIENDLY_POS)) {
        stop.friendlyPos = stopBaseObject->getBoolAttribute(SUMO_ATTR_FRIENDLY_POS);
    }
    // obtain posLat
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_POSITION_LAT)) {
        if (GNEAttributeCarrier::canParse<double>(stopBaseObject->getStringAttribute(SUMO_ATTR_POSITION_LAT))) {
            stop.posLat = GNEAttributeCarrier::parse<double>(stopBaseObject->getStringAttribute(SUMO_ATTR_POSITION_LAT));
            stop.parametersSet |= STOP_POSLAT_SET;
        } else {
            stop.posLat = INVALID_DOUBLE;
        }
    }
    // obtain actType
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_ACTTYPE)) {
        stop.actType = stopBaseObject->getStringAttribute(SUMO_ATTR_ACTTYPE);
    }
    // fill rest of parameters depending if it was edited
    if (stopBaseObject->hasTimeAttribute(SUMO_ATTR_DURATION)) {
        stop.duration = stopBaseObject->getTimeAttribute(SUMO_ATTR_DURATION);
        stop.parametersSet |= STOP_DURATION_SET;
    } else {
        stop.duration = -1;
    }
    if (stopBaseObject->hasTimeAttribute(SUMO_ATTR_UNTIL)) {
        stop.until = stopBaseObject->getTimeAttribute(SUMO_ATTR_UNTIL);
        stop.parametersSet |= STOP_UNTIL_SET;
    } else {
        stop.until = -1;
    }
    if (stopBaseObject->hasTimeAttribute(SUMO_ATTR_EXTENSION)) {
        stop.extension = stopBaseObject->getTimeAttribute(SUMO_ATTR_EXTENSION);
        stop.parametersSet |= STOP_EXTENSION_SET;
    }
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_TRIGGERED)) {
        if (stopBaseObject->getStringAttribute(SUMO_ATTR_TRIGGERED) == "person") {
            stop.parametersSet |= STOP_TRIGGER_SET;
            stop.triggered = true;
        } else if (stopBaseObject->getStringAttribute(SUMO_ATTR_TRIGGERED) == "container") {
            stop.parametersSet |= STOP_TRIGGER_SET;
            stop.parametersSet |= STOP_CONTAINER_TRIGGER_SET;
            stop.containerTriggered = true;
        } else if (stopBaseObject->getStringAttribute(SUMO_ATTR_TRIGGERED) == "join") {
            stop.parametersSet |= STOP_TRIGGER_SET;
            stop.joinTriggered = true;
        }
    }
    if (stopBaseObject->hasStringListAttribute(SUMO_ATTR_EXPECTED)) {
        const auto expected = stopBaseObject->getStringListAttribute(SUMO_ATTR_EXPECTED);
        if (expected.size() > 0) {
            if (stop.triggered) {
                for (const auto& id : expected) {
                    stop.awaitedPersons.insert(id);
                }
                stop.parametersSet |= STOP_EXPECTED_SET;
            } else if (stop.containerTriggered) {
                for (const auto& id : expected) {
                    stop.awaitedContainers.insert(id);
                }
                stop.parametersSet |= STOP_EXPECTED_CONTAINERS_SET;
            }
        }
    }
    if (stopBaseObject->hasStringListAttribute(SUMO_ATTR_PERMITTED)) {
        const auto permitted = stopBaseObject->getStringListAttribute(SUMO_ATTR_PERMITTED);
        if (permitted.size() > 0) {
            stop.parametersSet |= STOP_PERMITTED_SET;
            for (const auto& permit : permitted) {
                stop.permitted.insert(permit);
            }
        }
    }
    if (stopBaseObject->hasBoolAttribute(SUMO_ATTR_PARKING)) {
        if (stopBaseObject->getBoolAttribute(SUMO_ATTR_PARKING)) {
            stop.parametersSet |= STOP_PARKING_SET;
        }
    }
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_TRIP_ID)) {
        stop.tripId = stopBaseObject->getStringAttribute(SUMO_ATTR_TRIP_ID);
        stop.parametersSet |= STOP_TRIP_ID_SET;
    }
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_LINE)) {
        stop.line = stopBaseObject->getStringAttribute(SUMO_ATTR_LINE);
        stop.parametersSet |= STOP_LINE_SET;
    }
    if (stopBaseObject->hasDoubleAttribute(SUMO_ATTR_SPEED) && (stopBaseObject->getDoubleAttribute(SUMO_ATTR_SPEED) > 0)) {
        stop.speed = stopBaseObject->getDoubleAttribute(SUMO_ATTR_SPEED);
        stop.parametersSet |= STOP_SPEED_SET;
    }
    if (stopBaseObject->hasStringAttribute(SUMO_ATTR_INDEX)) {
        if (stopBaseObject->getStringAttribute(SUMO_ATTR_INDEX) == "fit") {
            stop.index = STOP_INDEX_FIT;
        } else if (stopBaseObject->getStringAttribute(SUMO_ATTR_INDEX) == "end") {
            stop.index = STOP_INDEX_END;
        } else if (GNEAttributeCarrier::canParse<int>(stopBaseObject->getStringAttribute(SUMO_ATTR_INDEX))) {
            stop.index = GNEAttributeCarrier::parse<int>(stopBaseObject->getStringAttribute(SUMO_ATTR_INDEX));
        } else {
            stop.index = STOP_INDEX_END;
        }
    } else {
        stop.index = STOP_INDEX_END;
    }
    // refresh stop attributes
    myStopAttributes->refreshAttributesCreator();
    // set tag
    stopBaseObject->setTag(stopTag);
    stopBaseObject->setStopParameter(stop);
    return true;
}


DemandElementSelector*
GNEStopFrame::getStopParentSelector() const {
    return myStopParentSelector;
}

// ===========================================================================
// protected
// ===========================================================================

void
GNEStopFrame::tagSelected() {
    if (myStopTagSelector->getCurrentTemplateAC()) {
        // show Stop type selector module
        myStopAttributes->showAttributesCreatorModule(myStopTagSelector->getCurrentTemplateAC(), {});
        myNeteditAttributes->showNeteditAttributesModule(myStopTagSelector->getCurrentTemplateAC());
        myHelpCreation->showHelpCreation();
    } else {
        // hide all modules if stop parent isn't valid
        myStopAttributes->hideAttributesCreatorModule();
        myNeteditAttributes->hideNeteditAttributesModule();
        myHelpCreation->hideHelpCreation();
    }
}


void
GNEStopFrame::demandElementSelected() {
    // show or hide modules depending if current selected stop parent is valid
    if (myStopParentSelector->getCurrentDemandElement()) {
        myStopTagSelector->showTagSelector();
        if (myStopTagSelector->getCurrentTemplateAC()) {
            // show modules
            myStopAttributes->showAttributesCreatorModule(myStopTagSelector->getCurrentTemplateAC(), {});
            myNeteditAttributes->showNeteditAttributesModule(myStopTagSelector->getCurrentTemplateAC());
            myHelpCreation->showHelpCreation();
        } else {
            myStopAttributes->hideAttributesCreatorModule();
            myNeteditAttributes->hideNeteditAttributesModule();
            myHelpCreation->hideHelpCreation();
        }
    } else {
        // hide modules
        myStopTagSelector->hideTagSelector();
        myStopAttributes->hideAttributesCreatorModule();
        myNeteditAttributes->hideNeteditAttributesModule();
        myHelpCreation->hideHelpCreation();
    }
}


/****************************************************************************/
