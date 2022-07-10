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
/// @file    GNEStop.cpp
/// @author  Pablo Alvarez Lopez
/// @date    March 2019
///
// Representation of Stops in NETEDIT
/****************************************************************************/
#include <cmath>
#include <netedit/GNENet.h>
#include <netedit/GNEUndoList.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNEViewParent.h>
#include <netedit/changes/GNEChange_Attribute.h>
#include <netedit/changes/GNEChange_EnableAttribute.h>
#include <netedit/frames/common/GNEMoveFrame.h>
#include <netedit/frames/demand/GNEStopFrame.h>
#include <netedit/frames/GNEDemandSelector.h>
#include <utils/gui/div/GLHelper.h>
#include <utils/gui/globjects/GLIncludes.h>
#include <utils/vehicle/SUMORouteHandler.h>

#include "GNEStop.h"


// ===========================================================================
// member method definitions
// ===========================================================================

GNEStop::GNEStop(SumoXMLTag tag, GNENet* net) :
    GNEDemandElement("", net, GLO_STOP, tag, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {}, {}, {}),
myCreationIndex(myNet->getAttributeCarriers()->getStopIndex()) {
    // reset default values
    resetDefaultValues();
    // enable parking for stops in parkin)gAreas
    if ((tag == SUMO_TAG_STOP_PARKINGAREA) || (tag == GNE_TAG_WAYPOINT_PARKINGAREA)) {
        parametersSet |= STOP_PARKING_SET;
    }
    // set flags
    parking = (parametersSet & STOP_PARKING_SET);
    // set waypoint speed
    myTagProperty.isWaypoint() ? parametersSet |= STOP_SPEED_SET : parametersSet &= ~STOP_SPEED_SET;
}


GNEStop::GNEStop(SumoXMLTag tag, GNENet* net, GNEDemandElement* stopParent, GNEAdditional* stoppingPlace, const SUMOVehicleParameter::Stop& stopParameter) :
    GNEDemandElement(stopParent, net, GLO_STOP, tag, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {stoppingPlace}, {stopParent}, {}),
SUMOVehicleParameter::Stop(stopParameter),
myCreationIndex(myNet->getAttributeCarriers()->getStopIndex()) {
    // enable parking for stops in parkingAreas
    if ((tag == SUMO_TAG_STOP_PARKINGAREA) || (tag == GNE_TAG_WAYPOINT_PARKINGAREA)) {
        parametersSet |= STOP_PARKING_SET;
    }
    // set flags
    parking = (parametersSet & STOP_PARKING_SET);
    // set tripID and line
    (stopParameter.tripId.size() > 0) ? parametersSet |= STOP_TRIP_ID_SET : parametersSet &= ~STOP_TRIP_ID_SET;
    (stopParameter.line.size() > 0) ? parametersSet |= STOP_LINE_SET : parametersSet &= ~STOP_LINE_SET;
    // set waypoint speed
    myTagProperty.isWaypoint() ? parametersSet |= STOP_SPEED_SET : parametersSet &= ~STOP_SPEED_SET;
}


GNEStop::GNEStop(SumoXMLTag tag, GNENet* net, GNEDemandElement* stopParent, GNELane* lane, const SUMOVehicleParameter::Stop& stopParameter) :
    GNEDemandElement(stopParent, net, GLO_STOP, tag, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {lane}, {}, {stopParent}, {}),
SUMOVehicleParameter::Stop(stopParameter),
myCreationIndex(myNet->getAttributeCarriers()->getStopIndex()) {
    // set flags
    parking = (parametersSet & STOP_PARKING_SET);
    // set tripID and line
    (stopParameter.tripId.size() > 0) ? parametersSet |= STOP_TRIP_ID_SET : parametersSet &= ~STOP_TRIP_ID_SET;
    (stopParameter.line.size() > 0) ? parametersSet |= STOP_LINE_SET : parametersSet &= ~STOP_LINE_SET;
    // set waypoint speed
    myTagProperty.isWaypoint() ? parametersSet |= STOP_SPEED_SET : parametersSet &= ~STOP_SPEED_SET;
}


GNEStop::GNEStop(SumoXMLTag tag, GNENet* net, GNEDemandElement* stopParent, GNEEdge* edge, const SUMOVehicleParameter::Stop& stopParameter) :
    GNEDemandElement(stopParent, net, GLO_STOP, tag, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {edge}, {}, {}, {stopParent}, {}),
SUMOVehicleParameter::Stop(stopParameter),
myCreationIndex(myNet->getAttributeCarriers()->getStopIndex()) {
    // enable parking for stops in parkingAreas
    if ((tag == SUMO_TAG_STOP_PARKINGAREA) || (tag == GNE_TAG_WAYPOINT_PARKINGAREA)) {
        parametersSet |= STOP_PARKING_SET;
    }
    // set flags
    parking = (parametersSet & STOP_PARKING_SET);
    triggered = (parametersSet & STOP_TRIGGER_SET);
    containerTriggered = (parametersSet & STOP_CONTAINER_TRIGGER_SET);
    joinTriggered = (parametersSet & STOP_JOIN_SET);
    // set tripID and line
    (stopParameter.tripId.size() > 0) ? parametersSet |= STOP_TRIP_ID_SET : parametersSet &= ~STOP_TRIP_ID_SET;
    (stopParameter.line.size() > 0) ? parametersSet |= STOP_LINE_SET : parametersSet &= ~STOP_LINE_SET;
    // set waypoint speed
    myTagProperty.isWaypoint() ? parametersSet |= STOP_SPEED_SET : parametersSet &= ~STOP_SPEED_SET;
}


GNEStop::~GNEStop() {}


GNEMoveOperation*
GNEStop::getMoveOperation() {
    if ((myTagProperty.getTag() == GNE_TAG_STOPPERSON_EDGE) || (myTagProperty.getTag() == GNE_TAG_STOPCONTAINER_EDGE)) {
        // return move operation for additional placed over shape
        return new GNEMoveOperation(this, getParentEdges().front()->getLanes().front(), endPos, false);
    } else if ((myTagProperty.getTag() == SUMO_TAG_STOP_LANE) || (myTagProperty.getTag() == GNE_TAG_WAYPOINT_LANE)) {
        // get allow change lane
        const bool allowChangeLane = myNet->getViewNet()->getViewParent()->getMoveFrame()->getCommonModeOptions()->getAllowChangeLane();
        // fist check if we're moving only extremes
        if (myNet->getViewNet()->getEditModes().isCurrentSupermodeDemand() &&
                (myNet->getViewNet()->getEditModes().demandEditMode == DemandEditMode::DEMAND_MOVE) &&
                myNet->getViewNet()->getMouseButtonKeyPressed().shiftKeyPressed()) {
            // get snap radius
            const double snap_radius = myNet->getViewNet()->getVisualisationSettings().neteditSizeSettings.additionalGeometryPointRadius;
            // get mouse position
            const Position mousePosition = myNet->getViewNet()->getPositionInformation();
            // check if we clicked over start or end position
            if ((startPos != INVALID_DOUBLE) && (myDemandElementGeometry.getShape().front().distanceSquaredTo2D(mousePosition) <= (snap_radius * snap_radius))) {
                // move only start position
                return new GNEMoveOperation(this, getParentLanes().front(), startPos, getParentLanes().front()->getLaneShape().length2D() - POSITION_EPS,
                                            allowChangeLane, GNEMoveOperation::OperationType::ONE_LANE_MOVEFIRST);
            } else if ((endPos != INVALID_DOUBLE) && (myDemandElementGeometry.getShape().back().distanceSquaredTo2D(mousePosition) <= (snap_radius * snap_radius))) {
                // move only end position
                return new GNEMoveOperation(this, getParentLanes().front(), 0, endPos,
                                            allowChangeLane, GNEMoveOperation::OperationType::ONE_LANE_MOVESECOND);
            } else {
                return nullptr;
            }
        } else if ((startPos != INVALID_DOUBLE) && (endPos != INVALID_DOUBLE)) {
            // move both start and end positions
            return new GNEMoveOperation(this, getParentLanes().front(), startPos, endPos,
                                        allowChangeLane, GNEMoveOperation::OperationType::ONE_LANE_MOVEBOTH);
        } else if (startPos != INVALID_DOUBLE) {
            // move only start position
            return new GNEMoveOperation(this, getParentLanes().front(), startPos, getParentLanes().front()->getLaneShape().length2D() - POSITION_EPS,
                                        allowChangeLane, GNEMoveOperation::OperationType::ONE_LANE_MOVEFIRST);
        } else if (startPos != INVALID_DOUBLE) {
            // move only end position
            return new GNEMoveOperation(this, getParentLanes().front(), 0, endPos,
                                        allowChangeLane, GNEMoveOperation::OperationType::ONE_LANE_MOVESECOND);
        } else {
            // start and end positions undefined, then nothing to move
            return nullptr;
        }
    } else {
        return nullptr;
    }
}


std::string
GNEStop::getBegin() const {
    return "";
}


void
GNEStop::writeDemandElement(OutputDevice& device) const {
    device.openTag(SUMO_TAG_STOP);
    if (getParentAdditionals().size() > 0) {
        if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_BUS_STOP) {
            device.writeAttr(SUMO_ATTR_BUS_STOP, getParentAdditionals().front()->getID());
        }
        if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_CONTAINER_STOP) {
            device.writeAttr(SUMO_ATTR_CONTAINER_STOP, getParentAdditionals().front()->getID());
        }
        if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_CHARGING_STATION) {
            device.writeAttr(SUMO_ATTR_CHARGING_STATION, getParentAdditionals().front()->getID());
        }
        if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_PARKING_AREA) {
            device.writeAttr(SUMO_ATTR_PARKING_AREA, getParentAdditionals().front()->getID());
        }
    } else {
        if (getParentLanes().size() > 0) {
            device.writeAttr(SUMO_ATTR_LANE, getParentLanes().front()->getID());
        } else {
            device.writeAttr(SUMO_ATTR_EDGE, getParentEdges().front()->getID());
        }
        if ((parametersSet & STOP_START_SET) != 0) {
            device.writeAttr(SUMO_ATTR_STARTPOS, startPos);
        }
        if ((parametersSet & STOP_END_SET) != 0) {
            device.writeAttr(SUMO_ATTR_ENDPOS, endPos);
        }
    }
    // write rest of attributes
    write(device, true, false);
}


GNEDemandElement::Problem
GNEStop::isDemandElementValid() const {
    if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
        // get lane
        const GNELane* firstLane = getFirstAllowedLane();
        // only Stops placed over lanes can be invalid
        if (myTagProperty.getTag() != SUMO_TAG_STOP_LANE) {
            return isPersonPlanValid();
        } else if (friendlyPos) {
            // with friendly position enabled position are "always fixed"
            return isPersonPlanValid();
        } else if (firstLane != nullptr) {
            // obtain lane length
            const double laneLength = getParentEdges().front()->getNBEdge()->getFinalLength() * firstLane->getLengthGeometryFactor();
            // declare end pos fixed
            const double endPosFixed = (endPos < 0) ? (endPos + laneLength) : endPos;
            // check values
            if ((endPosFixed <= getParentEdges().front()->getNBEdge()->getFinalLength()) && (endPosFixed > 0)) {
                return isPersonPlanValid();
            } else {
                return Problem::INVALID_STOPPOSITION;
            }
        } else {
            return Problem::INVALID_ELEMENT;
        }
    } else {
        // only Stops placed over lanes can be invalid
        if (myTagProperty.getTag() != SUMO_TAG_STOP_LANE) {
            return Problem::OK;
        } else if (friendlyPos) {
            // with friendly position enabled position are "always fixed"
            return Problem::OK;
        } else {
            // obtain lane length
            double laneLength = getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength() * getParentLanes().front()->getLengthGeometryFactor();
            // declare a copy of start and end positions
            double startPosCopy = startPos;
            double endPosCopy = endPos;
            // check if position has to be fixed
            if (startPosCopy < 0) {
                startPosCopy += laneLength;
            }
            if (endPosCopy < 0) {
                endPosCopy += laneLength;
            }
            // check values
            if ((startPosCopy >= 0) && (endPosCopy <= getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength()) && ((endPosCopy - startPosCopy) >= POSITION_EPS)) {
                return Problem::OK;
            } else {
                return Problem::INVALID_STOPPOSITION;
            }
        }
    }
}


std::string
GNEStop::getDemandElementProblem() const {
    if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
        if (friendlyPos) {
            return getPersonPlanProblem();
        } else {
            // obtain lane length
            const double laneLength = getParentEdges().front()->getNBEdge()->getFinalLength();
            // declare end pos fixed
            const double endPosFixed = (endPos < 0) ? (endPos + laneLength) : endPos;
            // check positions over lane
            if (endPosFixed < 0) {
                return (toString(SUMO_ATTR_ENDPOS) + " < 0");
            } else if (endPosFixed > getParentEdges().front()->getNBEdge()->getFinalLength()) {
                return (toString(SUMO_ATTR_ENDPOS) + " > lanes's length");
            } else {
                return getPersonPlanProblem();
            }
        }
    } else {
        // declare a copy of start and end positions
        double startPosCopy = startPos;
        double endPosCopy = endPos;
        // obtain lane length
        double laneLength = getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength();
        // check if position has to be fixed
        if (startPosCopy < 0) {
            startPosCopy += laneLength;
        }
        if (endPosCopy < 0) {
            endPosCopy += laneLength;
        }
        // declare variables
        std::string errorStart, separator, errorEnd;
        // check positions over lane
        if (startPosCopy < 0) {
            errorStart = (toString(SUMO_ATTR_STARTPOS) + " < 0");
        } else if (startPosCopy > getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength()) {
            errorStart = (toString(SUMO_ATTR_STARTPOS) + " > lanes's length");
        }
        if (endPosCopy < 0) {
            errorEnd = (toString(SUMO_ATTR_ENDPOS) + " < 0");
        } else if (endPosCopy > getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength()) {
            errorEnd = (toString(SUMO_ATTR_ENDPOS) + " > lanes's length");
        }
        // check separator
        if ((errorStart.size() > 0) && (errorEnd.size() > 0)) {
            separator = " and ";
        }
        return errorStart + separator + errorEnd;
    }
}


void
GNEStop::fixDemandElementProblem() {
    //
}


SUMOVehicleClass
GNEStop::getVClass() const {
    return getParentDemandElements().front()->getVClass();
}


const RGBColor&
GNEStop::getColor() const {
    if (getTagProperty().isPersonPlan() || getTagProperty().isContainerPlan()) {
        return myNet->getViewNet()->getVisualisationSettings().colorSettings.stopPersonColor;
    } else if (myNet->getViewNet()->getInspectedAttributeCarriers().size() > 0) {
        // get inspected AC
        const auto AC = myNet->getViewNet()->getInspectedAttributeCarriers().front();
        // check if is a route or a vehicle
        if ((AC->getTagProperty().isRoute() || AC->getTagProperty().isVehicle()) && (AC != getParentDemandElements().front())) {
            return RGBColor::GREY;
        }
    } else if (myNet->getViewNet()->getViewParent()->getStopFrame()->shown()) {
        if (myNet->getViewNet()->getViewParent()->getStopFrame()->getStopParentSelector()->getCurrentDemandElement() != getParentDemandElements().front()) {
            return RGBColor::GREY;
        }
    }
    // return default color
    if (myTagProperty.isWaypoint()) {
        return myNet->getViewNet()->getVisualisationSettings().colorSettings.waypointColor;
    } else {
        return myNet->getViewNet()->getVisualisationSettings().colorSettings.stopColor;
    }
}


void
GNEStop::updateGeometry() {
    // update geometry depending of parent
    if (getParentLanes().size() > 0) {
        // Cut shape using as delimitators fixed start position and fixed end position
        myDemandElementGeometry.updateGeometry(getParentLanes().front()->getLaneShape(), getStartGeometryPositionOverLane(), getEndGeometryPositionOverLane(), myMoveElementLateralOffset);
    } else if (getParentAdditionals().size() > 0) {
        if (getTagProperty().isPersonPlan() || getTagProperty().isContainerPlan()) {
            // get busStop shape
            const PositionVector& busStopShape = getParentAdditionals().front()->getAdditionalGeometry().getShape();
            // update demand element geometry using both positions
            myDemandElementGeometry.updateGeometry(busStopShape, busStopShape.length2D() - 0.6, busStopShape.length2D(), 0);
        } else {
            // use geometry of additional (busStop)
            myDemandElementGeometry = getParentAdditionals().at(0)->getAdditionalGeometry();
        }
    } else if (getParentEdges().size() > 0) {
        // get front and back lane
        const GNELane* frontLane = getParentEdges().front()->getLanes().front();
        const GNELane* backLane = getParentEdges().front()->getLanes().back();
        // get lane drawing constants
        GNELane::LaneDrawingConstants laneDrawingConstantsFront(myNet->getViewNet()->getVisualisationSettings(), frontLane);
        GNELane::LaneDrawingConstants laneDrawingConstantBack(myNet->getViewNet()->getVisualisationSettings(), backLane);
        // calculate front position
        const Position frontPosition = frontLane->getLaneShape().positionAtOffset2D(getAttributeDouble(SUMO_ATTR_ARRIVALPOS), laneDrawingConstantsFront.halfWidth);
        // calulate length between both shapes
        const double length = backLane->getLaneShape().distance2D(frontPosition, true);
        // calculate back position
        const Position backPosition = frontLane->getLaneShape().positionAtOffset2D(getAttributeDouble(SUMO_ATTR_ARRIVALPOS), (length + laneDrawingConstantBack.halfWidth - laneDrawingConstantsFront.halfWidth) * -1);
        // update demand element geometry using both positions
        myDemandElementGeometry.updateGeometry({frontPosition, backPosition});
    }
    /*
        // recompute geometry of all Demand elements related with this this stop
        if (getParentDemandElements().front()->getTagProperty().isRoute()) {
            getParentDemandElements().front()->updateGeometry();
        }
    */
}


Position
GNEStop::getPositionInView() const {
    if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
        // check if is placed over a busStop
        if (getParentAdditionals().size() > 0) {
            return getParentAdditionals().front()->getPositionInView();
        } else {
            // get lane
            const GNELane* personLane = getParentEdges().front()->getLaneByAllowedVClass(SVC_PEDESTRIAN);
            // get position over lane shape
            if (endPos <= 0) {
                return personLane->getLaneShape().front();
            } else if (endPos >= personLane->getLaneShape().length2D()) {
                return personLane->getLaneShape().back();
            } else {
                return personLane->getLaneShape().positionAtOffset2D(endPos);
            }
        }
    } else {
        if (getParentLanes().size() > 0) {
            return getParentLanes().front()->getLaneShape().positionAtOffset((startPos + endPos) / 2.0);
        } else if (getParentAdditionals().size() > 0) {
            return getParentAdditionals().front()->getPositionInView();
        } else {
            throw ProcessError("Invalid Stop parent");
        }
    }
}


std::string
GNEStop::getParentName() const {
    if (getParentDemandElements().size() > 0) {
        return getParentDemandElements().front()->getID();
    } else if (getParentAdditionals().size() > 0) {
        return getParentAdditionals().front()->getID();
    } else if (getParentLanes().size() > 0) {
        return getParentLanes().front()->getID();
    } else {
        throw ProcessError("Invalid parent");
    }
}


double
GNEStop::getExaggeration(const GUIVisualizationSettings& s) const {
    return s.addSize.getExaggeration(s, this);
}


Boundary
GNEStop::getCenteringBoundary() const {
    Boundary b;
    // Return Boundary depending if myMovingGeometryBoundary is initialised (important for move geometry)
    if (getParentAdditionals().size() > 0) {
        return getParentAdditionals().at(0)->getCenteringBoundary();
    } else if (myMovingGeometryBoundary.isInitialised()) {
        return myMovingGeometryBoundary;
    } else if (myDemandElementGeometry.getShape().size() > 0) {
        b = myDemandElementGeometry.getShape().getBoxBoundary();
    } else {
        b.add(getPositionInView());
    }
    b.grow(20);
    return b;
}


void
GNEStop::splitEdgeGeometry(const double /*splitPosition*/, const GNENetworkElement* /*originalElement*/, const GNENetworkElement* /*newElement*/, GNEUndoList* /*undoList*/) {
    // geometry of this element cannot be splitted
}


void
GNEStop::drawGL(const GUIVisualizationSettings& s) const {
    const bool draw = (getTagProperty().isStopPerson() || getTagProperty().isStopContainer()) ? drawPersonPlan() : canDrawVehicleStop();
    // check if stop can be drawn
    if (draw) {
        // Obtain exaggeration of the draw
        const double exaggeration = getExaggeration(s);
        // check if draw an stop for person/containers or for vehicles/routes
        if (getTagProperty().isStopPerson() || getTagProperty().isStopContainer()) {
            // check if draw stopPerson over busStop oder over lane
            if (getParentAdditionals().size() > 0) {
                drawStopPersonOverBusStop(s, exaggeration);
            } else {
                drawStopPersonOverEdge(s, exaggeration);
            }
            // draw person parent if this stop if their first person plan child
            if ((getParentDemandElements().size() == 1) && getParentDemandElements().front()->getChildDemandElements().front() == this) {
                getParentDemandElements().front()->drawGL(s);
            }
        } else {
            // draw vehicle over stop
            drawVehicleStop(s, exaggeration);
        }
        // Draw name
        drawName(getCenteringBoundary().getCenter(), s.scale, s.addName);
    }
}


void
GNEStop::computePathElement() {
    // only update geometry
    updateGeometry();
}


void
GNEStop::drawPartialGL(const GUIVisualizationSettings& /*s*/, const GNELane* /*lane*/, const GNEPathManager::Segment* /*segment*/, const double /*offsetFront*/) const {
    // Stops don't use drawPartialGL
}


void
GNEStop::drawPartialGL(const GUIVisualizationSettings& /*s*/, const GNELane* /*fromLane*/, const GNELane* /*toLane*/, const GNEPathManager::Segment* /*segment*/, const double /*offsetFront*/) const {
    // Stops don't use drawPartialGL
}


GNELane*
GNEStop::getFirstPathLane() const {
    // check if stop is placed over a busStop
    if (getParentAdditionals().size() > 0) {
        return getParentAdditionals().front()->getParentLanes().front();
    } else if (getParentEdges().size() > 0) {
        return getParentEdges().front()->getLaneByAllowedVClass(SVC_PEDESTRIAN);
    } else {
        return getParentLanes().front();
    }
}


GNELane*
GNEStop::getLastPathLane() const {
    // first and last path lane are the same
    return getFirstPathLane();
}


std::string
GNEStop::getAttribute(SumoXMLAttr key) const {
    switch (key) {
        case SUMO_ATTR_DURATION:
            if (isAttributeEnabled(key)) {
                return time2string(duration);
            } else {
                return "";
            }
        case SUMO_ATTR_UNTIL:
            if (isAttributeEnabled(key)) {
                return time2string(until);
            } else {
                return "";
            }
        case SUMO_ATTR_EXTENSION:
            if (isAttributeEnabled(key)) {
                return time2string(extension);
            } else {
                return "";
            }
        case SUMO_ATTR_TRIGGERED:
            if ((parametersSet & STOP_TRIGGER_SET) == false) {
                return "false";
            } else if (triggered) {
                return "person";
            } else if (containerTriggered) {
                return "container";
            } else {
                return "join";
            }
        case SUMO_ATTR_EXPECTED:
            if ((parametersSet & STOP_TRIGGER_SET) == false) {
                return "";
            } else if (triggered) {
                return toString(awaitedPersons);
            } else if (containerTriggered) {
                return toString(awaitedContainers);
            } else {
                return "";
            }
        case SUMO_ATTR_PERMITTED:
            return toString(permitted);
        case SUMO_ATTR_PARKING:
            if (parametersSet & STOP_PARKING_SET) {
                return "true";
            } else {
                return "false";
            }
        case SUMO_ATTR_ACTTYPE:
            return actType;
        case SUMO_ATTR_TRIP_ID:
            return tripId;
        case SUMO_ATTR_LINE:
            return line;
        // only for waypoints
        case SUMO_ATTR_SPEED:
            return toString(speed);
        // specific of Stops over stoppingPlaces
        case SUMO_ATTR_BUS_STOP:
        case SUMO_ATTR_CONTAINER_STOP:
        case SUMO_ATTR_CHARGING_STATION:
        case SUMO_ATTR_PARKING_AREA:
            return getParentAdditionals().front()->getID();
        // specific of stops over edges
        case SUMO_ATTR_EDGE:
            return getParentEdges().front()->getID();
        // specific of stops over lanes
        case SUMO_ATTR_LANE:
            return getParentLanes().front()->getID();
        case SUMO_ATTR_STARTPOS:
            return toString(startPos);
        case SUMO_ATTR_ENDPOS:
            return toString(endPos);
        case SUMO_ATTR_FRIENDLY_POS:
            return toString(friendlyPos);
        case SUMO_ATTR_POSITION_LAT:
            if (posLat == INVALID_DOUBLE) {
                return "";
            } else {
                return toString(posLat);
            }
        //
        case GNE_ATTR_SELECTED:
            return toString(isAttributeCarrierSelected());
        case GNE_ATTR_PARENT:
            return getParentDemandElements().front()->getID();
        case SUMO_ATTR_INDEX: {
            // extract all stops of demandElement parent
            std::vector<GNEDemandElement*> stops;
            for (const auto& parent : getParentDemandElements().front()->getChildDemandElements()) {
                if (parent->getTagProperty().isStop()) {
                    stops.push_back(parent);
                }
            }
            // find index in stops
            for (int i = 0; i < (int)stops.size(); i++) {
                if (stops.at(i) == this) {
                    return toString(i);
                }
            }
            return "invalid index";
        }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


double
GNEStop::getAttributeDouble(SumoXMLAttr key) const {
    switch (key) {
        case SUMO_ATTR_STARTPOS:
            if (getParentAdditionals().size() > 0) {
                return getParentAdditionals().front()->getAttributeDouble(SUMO_ATTR_STARTPOS);
            } else {
                return startPos;
            }
        case SUMO_ATTR_ENDPOS:
        case SUMO_ATTR_ARRIVALPOS:  // for person plans
            if (getParentAdditionals().size() > 0) {
                return getParentAdditionals().front()->getAttributeDouble(SUMO_ATTR_ENDPOS);
            } else {
                return endPos;
            }
        case SUMO_ATTR_INDEX: // for writting sorted
            return (double)myCreationIndex;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have a double attribute of type '" + toString(key) + "'");
    }
}


Position
GNEStop::getAttributePosition(SumoXMLAttr key) const {
    switch (key) {
        // we use SUMO_ATTR_ARRIVALPOS instead SUMO_ATTR_ENDPOS due it's a person plan
        case SUMO_ATTR_ARRIVALPOS: {
            if (getParentAdditionals().size() > 0) {
                // return first position of busStop
                return getParentAdditionals().front()->getAdditionalGeometry().getShape().front();
            } else {
                // get lane shape
                const PositionVector& laneShape = getLastPathLane()->getLaneShape();
                // continue depending of arrival position
                if (endPos == 0) {
                    return laneShape.front();
                } else if ((endPos == -1) || (endPos >= laneShape.length2D())) {
                    return laneShape.back();
                } else {
                    return laneShape.positionAtOffset2D(endPos);
                }
            }
        }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have a position attribute of type '" + toString(key) + "'");
    }
}


void
GNEStop::setAttribute(SumoXMLAttr key, const std::string& value, GNEUndoList* undoList) {
    if (value == getAttribute(key)) {
        return; //avoid needless changes, later logic relies on the fact that attributes have changed
    }
    switch (key) {
        case SUMO_ATTR_DURATION:
        case SUMO_ATTR_UNTIL:
        case SUMO_ATTR_EXTENSION:
        case SUMO_ATTR_TRIGGERED:
        case SUMO_ATTR_EXPECTED:
        case SUMO_ATTR_PERMITTED:
        case SUMO_ATTR_PARKING:
        case SUMO_ATTR_ACTTYPE:
        case SUMO_ATTR_TRIP_ID:
        case SUMO_ATTR_LINE:
        // only for waypoints
        case SUMO_ATTR_SPEED:
        // specific of Stops over stoppingPlaces
        case SUMO_ATTR_CHARGING_STATION:
        case SUMO_ATTR_PARKING_AREA:
        // specific of stops over lanes
        case SUMO_ATTR_LANE:
        case SUMO_ATTR_STARTPOS:
        case SUMO_ATTR_FRIENDLY_POS:
        case SUMO_ATTR_POSITION_LAT:
        //
        case GNE_ATTR_SELECTED:
        case GNE_ATTR_PARENT:
            undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            break;
        // special case for person plans
        case SUMO_ATTR_EDGE: {
            // get next personPlan
            GNEDemandElement* nextPersonPlan = getParentDemandElements().at(0)->getNextChildDemandElement(this);
            // continue depending of nextPersonPlan
            if (nextPersonPlan) {
                undoList->begin(myTagProperty.getGUIIcon(), "Change from attribute of next personPlan");
                nextPersonPlan->setAttribute(SUMO_ATTR_FROM, value, undoList);
                undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                undoList->end();
            } else {
                undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            }
            break;
        }
        case SUMO_ATTR_BUS_STOP:
            if (myTagProperty.isStopPerson()) {
                // get next person plan
                GNEDemandElement* nextPersonPlan = getParentDemandElements().at(0)->getNextChildDemandElement(this);
                // continue depending of nextPersonPlan
                if (nextPersonPlan) {
                    // obtain busStop
                    const GNEAdditional* busStop = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_BUS_STOP, value);
                    // change from attribute using edge ID
                    undoList->begin(myTagProperty.getGUIIcon(), "Change from attribute of next personPlan");
                    nextPersonPlan->setAttribute(SUMO_ATTR_FROM, busStop->getParentLanes().front()->getParentEdge()->getID(), undoList);
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                    undoList->end();
                } else {
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                }
            } else {
                undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            }
            break;
        case SUMO_ATTR_CONTAINER_STOP:
            if (myTagProperty.isStopContainer()) {
                // get next person plan
                GNEDemandElement* nextPersonPlan = getParentDemandElements().at(0)->getNextChildDemandElement(this);
                // continue depending of nextPersonPlan
                if (nextPersonPlan) {
                    // obtain containerStop
                    const GNEAdditional* containerStop = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CONTAINER_STOP, value);
                    // change from attribute using edge ID
                    undoList->begin(myTagProperty.getGUIIcon(), "Change from attribute of next personPlan");
                    nextPersonPlan->setAttribute(SUMO_ATTR_FROM, containerStop->getParentLanes().front()->getParentEdge()->getID(), undoList);
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                    undoList->end();
                } else {
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                }
            } else {
                undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            }
            break;
        case SUMO_ATTR_ENDPOS:
            if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
                // get previous person plan
                GNEDemandElement* previousPersonPlan = getParentDemandElements().at(0)->getPreviousChildDemandElement(this);
                // check if leave presonStop connected is enabled
                if (myNet->getViewNet()->getViewParent()->getMoveFrame()->getDemandModeOptions()->getLeaveStopPersonsConnected() &&
                        previousPersonPlan && previousPersonPlan->getTagProperty().hasAttribute(SUMO_ATTR_ARRIVALPOS)) {
                    // change from attribute using edge ID
                    undoList->begin(myTagProperty.getGUIIcon(), "Change arrivalPos attribute of previous personPlan");
                    previousPersonPlan->setAttribute(SUMO_ATTR_ARRIVALPOS, value, undoList);
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                    undoList->end();
                } else {
                    undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
                }
            } else {
                undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            }
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


bool
GNEStop::isValid(SumoXMLAttr key, const std::string& value) {
    // declare string error
    std::string error;
    switch (key) {
        case SUMO_ATTR_DURATION:
        case SUMO_ATTR_UNTIL:
        case SUMO_ATTR_EXTENSION:
            if (canParse<SUMOTime>(value)) {
                return parse<SUMOTime>(value) >= 0;
            } else {
                return false;
            }
        case SUMO_ATTR_TRIGGERED:
            if (value.empty()) {
                return false;
            } else {
                const std::set<std::string> expectedValues = {"true", "false", "person", "container", "join"};
                const std::vector<std::string> triggeredValues = parse<std::vector<std::string> >(value);
                for (const auto& triggeredValue : triggeredValues) {
                    if (expectedValues.find(triggeredValue) == expectedValues.end()) {
                        return false;
                    }
                }
                return true;
            }
        case SUMO_ATTR_EXPECTED:
            if (value.empty()) {
                return false;
            } else {
                const std::vector<std::string> expectedValues = parse<std::vector<std::string> >(value);
                for (const auto& expectedValue : expectedValues) {
                    if (!SUMOXMLDefinitions::isValidVehicleID(expectedValue)) {
                        return false;
                    }
                }
                return true;
            }
        case SUMO_ATTR_PERMITTED: {
            const std::vector<std::string> expectedValues = parse<std::vector<std::string> >(value);
            for (const auto& expectedValue : expectedValues) {
                if (!SUMOXMLDefinitions::isValidVehicleID(expectedValue)) {
                    return false;
                }
            }
            return true;
        }
        case SUMO_ATTR_PARKING:
            return canParse<bool>(value);
        case SUMO_ATTR_ACTTYPE:
            return true;
        case SUMO_ATTR_TRIP_ID:
            return SUMOXMLDefinitions::isValidVehicleID(value);
        case SUMO_ATTR_LINE:
            return true;
        // only for waypoints
        case SUMO_ATTR_SPEED:
            if (canParse<double>(value)) {
                return (parse<double>(value) >= 0);
            } else {
                return false;
            }
        // specific of Stops over stoppingPlaces
        case SUMO_ATTR_BUS_STOP:
            return (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_BUS_STOP, value, false) != nullptr);
        case SUMO_ATTR_CONTAINER_STOP:
            return (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CONTAINER_STOP, value, false) != nullptr);
        case SUMO_ATTR_CHARGING_STATION:
            return (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CHARGING_STATION, value, false) != nullptr);
        case SUMO_ATTR_PARKING_AREA:
            return (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_PARKING_AREA, value, false) != nullptr);
        // specific of stops over edges
        case SUMO_ATTR_EDGE:
            if (myNet->getAttributeCarriers()->retrieveEdge(value, false) != nullptr) {
                return true;
            } else {
                return false;
            }
        // specific of stops over lanes
        case SUMO_ATTR_LANE:
            if (myNet->getAttributeCarriers()->retrieveLane(value, false) != nullptr) {
                return true;
            } else {
                return false;
            }
        case SUMO_ATTR_STARTPOS:
            if (canParse<double>(value)) {
                return SUMORouteHandler::isStopPosValid(parse<double>(value), endPos, getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength(), POSITION_EPS, friendlyPos);
            } else {
                return false;
            }
        case SUMO_ATTR_ENDPOS:
            if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
                return canParse<double>(value) && fabs(parse<double>(value)) < getParentEdges().front()->getNBEdge()->getFinalLength();
            } else if (canParse<double>(value)) {
                return SUMORouteHandler::isStopPosValid(startPos, parse<double>(value), getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength(), POSITION_EPS, friendlyPos);
            } else {
                return false;
            }
        case SUMO_ATTR_FRIENDLY_POS:
            return canParse<bool>(value);
        case SUMO_ATTR_POSITION_LAT:
            if (value.empty()) {
                return true;
            } else {
                return canParse<double>(value);
            }
        //
        case GNE_ATTR_SELECTED:
            return canParse<bool>(value);
        case GNE_ATTR_PARENT:
            if (myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_PERSON, value, false) != nullptr) {
                return true;
            } else if (myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_PERSONFLOW, value, false) != nullptr) {
                return true;
            } else {
                return false;
            }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEStop::enableAttribute(SumoXMLAttr key, GNEUndoList* undoList) {
    switch (key) {
        case SUMO_ATTR_DURATION:
        case SUMO_ATTR_UNTIL:
        case SUMO_ATTR_EXTENSION:
        case SUMO_ATTR_EXPECTED:
        case SUMO_ATTR_EXPECTED_CONTAINERS:
            undoList->add(new GNEChange_EnableAttribute(this, key, true), true);
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEStop::disableAttribute(SumoXMLAttr key, GNEUndoList* undoList) {
    switch (key) {
        case SUMO_ATTR_DURATION:
        case SUMO_ATTR_UNTIL:
        case SUMO_ATTR_EXTENSION:
        case SUMO_ATTR_EXPECTED:
        case SUMO_ATTR_EXPECTED_CONTAINERS:
            undoList->add(new GNEChange_EnableAttribute(this, key, false), true);
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


bool
GNEStop::isAttributeEnabled(SumoXMLAttr key) const {
    switch (key) {
        // Currently stops parents cannot be edited
        case SUMO_ATTR_BUS_STOP:
        case SUMO_ATTR_CONTAINER_STOP:
        case SUMO_ATTR_CHARGING_STATION:
        case SUMO_ATTR_PARKING_AREA:
            return false;
        case SUMO_ATTR_DURATION:
            return (parametersSet & STOP_DURATION_SET) != 0;
        case SUMO_ATTR_UNTIL:
            return (parametersSet & STOP_UNTIL_SET) != 0;
        case SUMO_ATTR_EXTENSION:
            return (parametersSet & STOP_EXTENSION_SET) != 0;
        case SUMO_ATTR_EXPECTED:
            return (parametersSet & STOP_TRIGGER_SET) != 0;
        case SUMO_ATTR_PARKING:
            if (myTagProperty.getTag() == SUMO_TAG_STOP_PARKINGAREA) {
                return false;
            } else if (myTagProperty.getTag() == GNE_TAG_WAYPOINT_PARKINGAREA) {
                return false;
            } else {
                return true;
            }
        default:
            return true;
    }
}


std::string
GNEStop::getPopUpID() const {
    return getTagStr();
}


std::string
GNEStop::getHierarchyName() const {
    if (getParentAdditionals().size() > 0) {
        if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_BUS_STOP) {
            return "BusStop: " + getParentAdditionals().front()->getID();
        } else if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_CONTAINER_STOP) {
            return "containerStop: " + getParentAdditionals().front()->getID();
        } else if (getParentAdditionals().front()->getTagProperty().getTag() == SUMO_TAG_CHARGING_STATION) {
            return "chargingStation: " + getParentAdditionals().front()->getID();
        } else {
            return "parkingArea: " + getParentAdditionals().front()->getID();
        }
    } else if (getParentEdges().size() > 0) {
        return "edge: " + getParentEdges().front()->getID();
    } else if (getParentLanes().size() > 0) {
        return "lane: " + getParentLanes().front()->getID();
    } else {
        return "";
    }
}


const Parameterised::Map&
GNEStop::getACParametersMap() const {
    return getParametersMap();
}

double
GNEStop::getStartGeometryPositionOverLane() const {
    double fixedPos = startPos;
    const double len = getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength();
    if (fixedPos < 0) {
        fixedPos += len;
    }
    return fixedPos * getParentLanes().front()->getLengthGeometryFactor();
}


double
GNEStop::getEndGeometryPositionOverLane() const {
    double fixedPos = endPos;
    const double len = getParentLanes().front()->getParentEdge()->getNBEdge()->getFinalLength();
    if (fixedPos < 0) {
        fixedPos += len;
    }
    return fixedPos * getParentLanes().front()->getLengthGeometryFactor();
}

// ===========================================================================
// protected
// ===========================================================================

const GNELane*
GNEStop::getFirstAllowedLane() const {
    if (getParentEdges().empty()) {
        return nullptr;
    }
    for (const auto& pLane : getParentEdges().front()->getLanes()) {
        if (pLane->allowPedestrians()) {
            return pLane;
        }
    }
    return getParentEdges().front()->getLanes().front();
}


bool
GNEStop::canDrawVehicleStop() const {
    if (isAttributeCarrierSelected()) {
        return true;
    } else if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
        return true;
    } else if (myNet->getViewNet()->isAttributeCarrierInspected(getParentDemandElements().front())) {
        return true;
    } else if (myNet->getViewNet()->getDemandViewOptions().showAllTrips()) {
        return true;
    } else if ((getParentDemandElements().front()->getTagProperty().getTag() == GNE_TAG_VEHICLE_WITHROUTE) ||
               (getParentDemandElements().front()->getTagProperty().getTag() == GNE_TAG_FLOW_WITHROUTE)) {
        if (myNet->getViewNet()->isAttributeCarrierInspected(getParentDemandElements().front()->getChildDemandElements().front())) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}


void
GNEStop::drawVehicleStop(const GUIVisualizationSettings& s, const double exaggeration) const {
    ;
    // declare value to save stop color
    const RGBColor stopColor = drawUsingSelectColor() ? s.colorSettings.selectedRouteColor : getColor();
    // get lane
    const auto& stopLane = getParentLanes().size() > 0 ? getParentLanes().front() : nullptr;
    // get lane width
    const double width = stopLane ? stopLane->getParentEdge()->getNBEdge()->getLaneWidth(stopLane->getIndex()) * 0.5 : exaggeration * 0.8;
    // Start drawing adding an gl identificator
    GLHelper::pushName(getGlID());
    // Add a layer matrix
    GLHelper::pushMatrix();
    // set Color
    GLHelper::setColor(stopColor);
    // Start with the drawing of the area traslating matrix to origin
    myNet->getViewNet()->drawTranslateFrontAttributeCarrier(this, getType());
    // draw depending of details
    if (s.drawDetail(s.detailSettings.stopsDetails, exaggeration) && stopLane) {
        // Draw top and bot lines using shape, shapeRotations, shapeLengths and value of exaggeration
        GLHelper::drawBoxLines(myDemandElementGeometry.getShape(),
                               myDemandElementGeometry.getShapeRotations(),
                               myDemandElementGeometry.getShapeLengths(),
                               exaggeration * 0.1, 0, width);
        GLHelper::drawBoxLines(myDemandElementGeometry.getShape(),
                               myDemandElementGeometry.getShapeRotations(),
                               myDemandElementGeometry.getShapeLengths(),
                               exaggeration * 0.1, 0, width * -1);
        // Add a detail matrix
        GLHelper::pushMatrix();
        // move to geometry front
        glTranslated(myDemandElementGeometry.getShape().back().x(), myDemandElementGeometry.getShape().back().y(), 0.1);
        // rotate
        if (myDemandElementGeometry.getShapeRotations().size() > 0) {
            glRotated(myDemandElementGeometry.getShapeRotations().back(), 0, 0, 1);
        }
        // move again
        glTranslated(0, exaggeration * 0.5, 0);
        // draw stop front
        GLHelper::drawBoxLine(Position(0, 0), 0, exaggeration * 0.5, width);
        // move to "S" position
        glTranslated(0, 1, 0.1);
        // only draw text if isn't being drawn for selecting
        if (s.drawForRectangleSelection) {
            GLHelper::setColor(stopColor);
            GLHelper::drawBoxLine(Position(0, 1), 0, 2, 1);
        } else if (s.drawDetail(s.detailSettings.stopsText, exaggeration)) {
            // draw "S" symbol
            GLHelper::drawText(myTagProperty.isWaypoint() ? "W" : "S", Position(), .1, 2.8, stopColor, 180);
            // move to subtitle position
            glTranslated(0, 1.4, 0);
            // draw subtitle depending of tag
            GLHelper::drawText("lane", Position(), .1, 1, stopColor, 180);
            // check if draw index
            if (drawIndex()) {
                // move to index position
                glTranslated(-2.1, -2.4, 0);
                glRotated(-90, 0, 0, 1);
                // draw index
                GLHelper::drawText(getAttribute(SUMO_ATTR_INDEX), Position(), .1, 1, stopColor, 180);
            }
        }
        // pop detail matrix
        GLHelper::popMatrix();
        // draw geometry points
        drawGeometryPoints(s, stopColor);
    } else {
        // Draw the area using shape, shapeRotations, shapeLengths and value of exaggeration taked from stoppingPlace parent
        GUIGeometry::drawGeometry(s, myNet->getViewNet()->getPositionInformation(), myDemandElementGeometry, width);
        // only draw text if isn't being drawn for selecting
        if (s.drawDetail(s.detailSettings.stopsText, exaggeration) && drawIndex()) {
            // Add a detail matrix
            GLHelper::pushMatrix();
            // move to geometry front
            glTranslated(myDemandElementGeometry.getShape().back().x(), myDemandElementGeometry.getShape().back().y(), 0.1);
            // rotate
            if (myDemandElementGeometry.getShapeRotations().size() > 0) {
                glRotated(myDemandElementGeometry.getShapeRotations().back(), 0, 0, 1);
            }
            // move to index position
            glTranslated(-1.4, exaggeration * 0.5, 0.1);
            glRotated(-90, 0, 0, 1);
            // draw index
            GLHelper::drawText(getAttribute(SUMO_ATTR_INDEX), Position(), .1, 1, stopColor, 180);
        }
        // pop detail matrix
        GLHelper::popMatrix();
    }
    // pop layer matrix
    GLHelper::popMatrix();
    // Pop name
    GLHelper::popName();
    // draw lock icon
    GNEViewNetHelper::LockIcon::drawLockIcon(this, getType(), getPositionInView(), exaggeration);
    // check if dotted contour has to be drawn
    if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::INSPECT, s, myDemandElementGeometry.getShape(),
                width, exaggeration, true, true);
    }
    if (myNet->getViewNet()->getFrontAttributeCarrier() == this) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::FRONT, s, myDemandElementGeometry.getShape(),
                width, exaggeration, true, true);
    }
}


void
GNEStop::drawStopPersonOverEdge(const GUIVisualizationSettings& s, const double exaggeration) const {
    // Start drawing adding an gl identificator
    GLHelper::pushName(getGlID());
    // Add layer matrix matrix
    GLHelper::pushMatrix();
    // translate to front
    myNet->getViewNet()->drawTranslateFrontAttributeCarrier(this, getType());
    // declare stop color
    const RGBColor stopColor = drawUsingSelectColor() ? s.colorSettings.selectedPersonPlanColor : s.colorSettings.stopColor;
    // declare central line color
    const RGBColor centralLineColor = drawUsingSelectColor() ? stopColor.changedBrightness(-32) : RGBColor::WHITE;
    // set base color
    GLHelper::setColor(stopColor);
    // Draw the area using shape, shapeRotations, shapeLengths and value of exaggeration
    GUIGeometry::drawGeometry(s, myNet->getViewNet()->getPositionInformation(), myDemandElementGeometry, 0.3 * exaggeration);
    // move to front
    glTranslated(0, 0, .1);
    // set central color
    GLHelper::setColor(centralLineColor);
    // Draw the area using shape, shapeRotations, shapeLengths and value of exaggeration
    GUIGeometry::drawGeometry(s, myNet->getViewNet()->getPositionInformation(), myDemandElementGeometry, 0.05 * exaggeration);
    // move to icon position and front
    glTranslated(myDemandElementGeometry.getShape().front().x(), myDemandElementGeometry.getShape().front().y(), .1);
    // rotate over lane
    GUIGeometry::rotateOverLane((myDemandElementGeometry.getShapeRotations().front() * -1) + 90);
    // move again
    glTranslated(0, s.additionalSettings.vaporizerSize * exaggeration, 0);
    // Draw icon depending of Route Probe is selected and if isn't being drawn for selecting
    if (!s.drawForRectangleSelection && s.drawDetail(s.detailSettings.laneTextures, exaggeration)) {
        // set color
        glColor3d(1, 1, 1);
        // rotate texture
        glRotated(180, 0, 0, 1);
        // draw texture
        if (drawUsingSelectColor()) {
            GUITexturesHelper::drawTexturedBox(GUITextureSubSys::getTexture(GUITexture::STOPPERSON_SELECTED), s.additionalSettings.vaporizerSize * exaggeration);
        } else {
            GUITexturesHelper::drawTexturedBox(GUITextureSubSys::getTexture(GUITexture::STOPPERSON), s.additionalSettings.vaporizerSize * exaggeration);
        }
    } else {
        // rotate
        glRotated(22.5, 0, 0, 1);
        // set stop color
        GLHelper::setColor(stopColor);
        // move matrix
        glTranslated(0, 0, 0);
        // draw filled circle
        GLHelper::drawFilledCircle(0.1 + s.additionalSettings.vaporizerSize, 8);
    }
    // pop layer matrix
    GLHelper::popMatrix();
    // Pop name
    GLHelper::popName();
    // draw lock icon
    GNEViewNetHelper::LockIcon::drawLockIcon(this, getType(), getPositionInView(), exaggeration);
    // check if dotted contours has to be drawn
    if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::INSPECT, s, myDemandElementGeometry.getShape(), 0.3,
                exaggeration, 1, 1);
    }
    if (myNet->getViewNet()->getFrontAttributeCarrier() == this) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::FRONT, s, myDemandElementGeometry.getShape(), 0.3,
                exaggeration, 1, 1);
    }
}


void
GNEStop::drawStopPersonOverBusStop(const GUIVisualizationSettings& s, const double exaggeration) const {
    // Start drawing adding an gl identificator
    GLHelper::pushName(getGlID());
    // Add layer matrix matrix
    GLHelper::pushMatrix();
    // translate to front
    myNet->getViewNet()->drawTranslateFrontAttributeCarrier(this, getType());
    // declare stop color
    const RGBColor stopColor = drawUsingSelectColor() ? s.colorSettings.selectedPersonPlanColor : s.colorSettings.stopColor;
    // set base color
    GLHelper::setColor(stopColor);
    // Draw the area using shape, shapeRotations, shapeLengths and value of exaggeration
    GUIGeometry::drawGeometry(s, myNet->getViewNet()->getPositionInformation(), myDemandElementGeometry, s.stoppingPlaceSettings.busStopWidth * exaggeration);
    // move to icon position and front
    glTranslated(myDemandElementGeometry.getShape().getLineCenter().x(), myDemandElementGeometry.getShape().getLineCenter().y(), .1);
    // rotate over lane
    GUIGeometry::rotateOverLane((myDemandElementGeometry.getShapeRotations().front() * -1) + 90);
    // move again
    glTranslated(s.stoppingPlaceSettings.busStopWidth * exaggeration * -2, 0, 0);
    // Draw icon depending of Route Probe is selected and if isn't being drawn for selecting
    if (!s.drawForRectangleSelection && s.drawDetail(s.detailSettings.laneTextures, exaggeration)) {
        // set color
        glColor3d(1, 1, 1);
        // rotate texture
        glRotated(-90, 0, 0, 1);
        // draw texture
        if (drawUsingSelectColor()) {
            GUITexturesHelper::drawTexturedBox(GUITextureSubSys::getTexture(GUITexture::STOPPERSON_SELECTED), s.additionalSettings.vaporizerSize * exaggeration);
        } else {
            GUITexturesHelper::drawTexturedBox(GUITextureSubSys::getTexture(GUITexture::STOPPERSON), s.additionalSettings.vaporizerSize * exaggeration);
        }
    } else {
        // rotate
        glRotated(22.5, 0, 0, 1);
        // set stop color
        GLHelper::setColor(stopColor);
        // move matrix
        glTranslated(0, 0, 0);
        // draw filled circle
        GLHelper::drawFilledCircle(0.1 + s.additionalSettings.vaporizerSize, 8);
    }
    // pop layer matrix
    GLHelper::popMatrix();
    // Pop name
    GLHelper::popName();
    // draw lock icon
    GNEViewNetHelper::LockIcon::drawLockIcon(this, getType(), getPositionInView(), exaggeration);
    // check if dotted contours has to be drawn
    if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::INSPECT, s, myDemandElementGeometry.getShape(), 0.3,
                exaggeration, 1, 1);
    }
    if (myNet->getViewNet()->getFrontAttributeCarrier() == this) {
        GUIDottedGeometry::drawDottedContourShape(GUIDottedGeometry::DottedContourType::FRONT, s, myDemandElementGeometry.getShape(), 0.3,
                exaggeration, 1, 1);
    }
}


bool
GNEStop::drawIndex() const {
    // get stop frame
    const auto stopFrame = myNet->getViewNet()->getViewParent()->getStopFrame();
    // check conditions
    if (myNet->getViewNet()->isAttributeCarrierInspected(getParentDemandElements().front())) {
        return true;
    } else if (stopFrame->shown() && (stopFrame->getStopParentSelector()->getCurrentDemandElement() == getParentDemandElements().front())) {
        return true;
    } else {
        return false;
    }
}

// ===========================================================================
// private
// ===========================================================================

void
GNEStop::setAttribute(SumoXMLAttr key, const std::string& value) {
    switch (key) {
        case SUMO_ATTR_DURATION:
            if (value.empty()) {
                toogleAttribute(key, false);
            } else {
                toogleAttribute(key, true);
                duration = string2time(value);
            }
            break;
        case SUMO_ATTR_UNTIL:
            if (value.empty()) {
                toogleAttribute(key, false);
            } else {
                toogleAttribute(key, true);
                until = string2time(value);
            }
            break;
        case SUMO_ATTR_EXTENSION:
            if (value.empty()) {
                toogleAttribute(key, false);
            } else {
                toogleAttribute(key, true);
                extension = string2time(value);
            }
            break;
        case SUMO_ATTR_TRIGGERED:
            // reset all flags
            triggered = false;
            containerTriggered = false;
            joinTriggered = false;
            // disable all flags
            parametersSet &= ~STOP_JOIN_SET;
            parametersSet &= ~STOP_TRIGGER_SET;
            parametersSet &= ~STOP_EXPECTED_SET;
            parametersSet &= ~STOP_CONTAINER_TRIGGER_SET;
            parametersSet &= ~STOP_EXPECTED_CONTAINERS_SET;
            // check value
            if (value == "person") {
                parametersSet |= STOP_TRIGGER_SET;
                triggered = true;
                if (awaitedPersons.size() > 0) {
                    parametersSet |= STOP_EXPECTED_SET;
                }
            } else if (value == "container") {
                parametersSet |= STOP_TRIGGER_SET;
                parametersSet |= STOP_CONTAINER_TRIGGER_SET;
                containerTriggered = true;
                if (awaitedPersons.size() > 0) {
                    parametersSet |= STOP_EXPECTED_CONTAINERS_SET;
                }
            } else if (value == "join") {
                parametersSet |= STOP_TRIGGER_SET;
                joinTriggered = true;
            }
            break;
        case SUMO_ATTR_EXPECTED:
            if (triggered) {
                awaitedPersons = parse<std::set<std::string> >(value);
                if (awaitedPersons.size() > 0) {
                    parametersSet |= STOP_EXPECTED_SET;
                } else {
                    parametersSet &= ~STOP_EXPECTED_SET;
                }
            } else if (containerTriggered) {
                awaitedContainers = parse<std::set<std::string> >(value);
                if (awaitedContainers.size() > 0) {
                    parametersSet |= STOP_EXPECTED_CONTAINERS_SET;
                } else {
                    parametersSet &= ~STOP_EXPECTED_CONTAINERS_SET;
                }
            }
            break;
        case SUMO_ATTR_PERMITTED:
            if (value.empty()) {
                parametersSet &= ~STOP_PERMITTED_SET;
            } else {
                parametersSet |= STOP_PERMITTED_SET;
                permitted = parse<std::set<std::string> >(value);
            }
            break;
        case SUMO_ATTR_PARKING:
            if (parse<bool>(value)) {
                parametersSet |= STOP_PARKING_SET;
            } else {
                parametersSet &= ~STOP_PARKING_SET;
            }
            // set flag
            parking = ((parametersSet & STOP_PARKING_SET) != 0);
            break;
        case SUMO_ATTR_ACTTYPE:
            actType = value;
            break;
        case SUMO_ATTR_TRIP_ID:
            tripId = value;
            toogleAttribute(key, (value.size() > 0));
            break;
        case SUMO_ATTR_LINE:
            line = value;
            toogleAttribute(key, (value.size() > 0));
            break;
        // only for waypoints
        case SUMO_ATTR_SPEED:
            speed = parse<double>(value);
            break;
        // specific of Stops over stoppingPlaces
        case SUMO_ATTR_BUS_STOP:
            replaceAdditionalParent(SUMO_TAG_BUS_STOP, value);
            updateGeometry();
            break;
        case SUMO_ATTR_CONTAINER_STOP:
            replaceAdditionalParent(SUMO_TAG_CONTAINER_STOP, value);
            updateGeometry();
            break;
        case SUMO_ATTR_CHARGING_STATION:
            replaceAdditionalParent(SUMO_TAG_CHARGING_STATION, value);
            updateGeometry();
            break;
        case SUMO_ATTR_PARKING_AREA:
            replaceAdditionalParent(SUMO_TAG_PARKING_AREA, value);
            updateGeometry();
            break;
        // specific of Stops over edges
        case SUMO_ATTR_EDGE:
            replaceDemandParentEdges(value);
            updateGeometry();
            edge = value;
            break;
        // specific of Stops over lanes
        case SUMO_ATTR_LANE:
            replaceDemandParentLanes(value);
            updateGeometry();
            break;
        case SUMO_ATTR_STARTPOS:
            startPos = parse<double>(value);
            updateGeometry();
            break;
        case SUMO_ATTR_ENDPOS:
            endPos = parse<double>(value);
            updateGeometry();
            break;
        case SUMO_ATTR_FRIENDLY_POS:
            friendlyPos = parse<bool>(value);
            break;
        case SUMO_ATTR_POSITION_LAT:
            if (value.empty()) {
                posLat = INVALID_DOUBLE;
                parametersSet &= ~STOP_POSLAT_SET;
            } else {
                posLat = parse<double>(value);
                parametersSet |= STOP_POSLAT_SET;
            }
            break;
        //
        case GNE_ATTR_SELECTED:
            if (parse<bool>(value)) {
                selectAttributeCarrier();
            } else {
                unselectAttributeCarrier();
            }
            break;
        case GNE_ATTR_PARENT:
            if (myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_PERSON, value, false) != nullptr) {
                replaceDemandElementParent(SUMO_TAG_PERSON, value, 0);
            } else if (myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_PERSONFLOW, value, false) != nullptr) {
                replaceDemandElementParent(SUMO_TAG_PERSONFLOW, value, 0);
            }
            updateGeometry();
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEStop::toogleAttribute(SumoXMLAttr key, const bool value) {
    switch (key) {
        case SUMO_ATTR_DURATION:
            if (value) {
                parametersSet |= STOP_DURATION_SET;
            } else {
                parametersSet &= ~STOP_DURATION_SET;
            }
            break;
        case SUMO_ATTR_UNTIL:
            if (value) {
                parametersSet |= STOP_UNTIL_SET;
            } else {
                parametersSet &= ~STOP_UNTIL_SET;
            }
            break;
        case SUMO_ATTR_EXTENSION:
            if (value) {
                parametersSet |= STOP_EXTENSION_SET;
            } else {
                parametersSet &= ~STOP_EXTENSION_SET;
            }
            break;
        case SUMO_ATTR_TRIP_ID:
            if (value) {
                parametersSet |= STOP_TRIP_ID_SET;
            } else {
                parametersSet &= ~STOP_TRIP_ID_SET;
            }
            break;
        case SUMO_ATTR_LINE:
            if (value) {
                parametersSet |= STOP_LINE_SET;
            } else {
                parametersSet &= ~STOP_LINE_SET;
            }
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEStop::setMoveShape(const GNEMoveResult& moveResult) {
    if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
        // change endPos
        endPos = moveResult.newFirstPos;
    } else {
        if (moveResult.operationType == GNEMoveOperation::OperationType::ONE_LANE_MOVEFIRST) {
            // change only start position
            startPos = moveResult.newFirstPos;
            // adjust startPos
            if (startPos > (getAttributeDouble(SUMO_ATTR_ENDPOS) - POSITION_EPS)) {
                startPos = (getAttributeDouble(SUMO_ATTR_ENDPOS) - POSITION_EPS);
            }
        } else if (moveResult.operationType == GNEMoveOperation::OperationType::ONE_LANE_MOVESECOND) {
            // change only end position
            endPos = moveResult.newFirstPos;
            // adjust endPos
            if (endPos < (getAttributeDouble(SUMO_ATTR_STARTPOS) + POSITION_EPS)) {
                endPos = (getAttributeDouble(SUMO_ATTR_STARTPOS) + POSITION_EPS);
            }
        } else {
            // change both position
            startPos = moveResult.newFirstPos;
            endPos = moveResult.newSecondPos;
            // set lateral offset
            myMoveElementLateralOffset = moveResult.firstLaneOffset;
        }
    }
    // update geometry
    updateGeometry();
}


void
GNEStop::commitMoveShape(const GNEMoveResult& moveResult, GNEUndoList* undoList) {
    // begin change attribute
    undoList->begin(myTagProperty.getGUIIcon(), "position of " + getTagStr());
    if (myTagProperty.isStopPerson() || myTagProperty.isStopContainer()) {
        // now adjust endPos position
        setAttribute(SUMO_ATTR_ENDPOS, toString(moveResult.newFirstPos), undoList);
    } else {
        // set attributes depending of operation type
        if (moveResult.operationType == GNEMoveOperation::OperationType::ONE_LANE_MOVEFIRST) {
            // set only start position
            undoList->changeAttribute(new GNEChange_Attribute(this, SUMO_ATTR_STARTPOS, toString(moveResult.newFirstPos)));
        } else if (moveResult.operationType == GNEMoveOperation::OperationType::ONE_LANE_MOVESECOND) {
            // set only end position
            undoList->changeAttribute(new GNEChange_Attribute(this, SUMO_ATTR_ENDPOS, toString(moveResult.newFirstPos)));
        } else {
            // set both
            undoList->changeAttribute(new GNEChange_Attribute(this, SUMO_ATTR_STARTPOS, toString(moveResult.newFirstPos)));
            undoList->changeAttribute(new GNEChange_Attribute(this, SUMO_ATTR_ENDPOS, toString(moveResult.newSecondPos)));
            // check if lane has to be changed
            if (moveResult.newFirstLane) {
                // set new lane
                setAttribute(SUMO_ATTR_LANE, moveResult.newFirstLane->getID(), undoList);
            }
        }
    }
    // end change attribute
    undoList->end();
}


void
GNEStop::drawGeometryPoints(const GUIVisualizationSettings& s, const RGBColor& baseColor) const {
    // first check that we're in move mode and shift key is pressed
    if (myNet->getViewNet()->getEditModes().isCurrentSupermodeDemand() &&
            (myNet->getViewNet()->getEditModes().demandEditMode == DemandEditMode::DEMAND_MOVE) &&
            myNet->getViewNet()->getMouseButtonKeyPressed().shiftKeyPressed()) {
        // calculate new color
        const RGBColor color = baseColor.changedBrightness(-50);
        // push matrix
        GLHelper::pushMatrix();
        // translated to front
        glTranslated(0, 0, 0.1);
        // set color
        GLHelper::setColor(color);
        // draw points
        if (startPos != INVALID_DOUBLE) {
            // push geometry point matrix
            GLHelper::pushMatrix();
            glTranslated(myDemandElementGeometry.getShape().front().x(), myDemandElementGeometry.getShape().front().y(), 0.1);
            // draw geometry point
            GLHelper::drawFilledCircle(s.neteditSizeSettings.additionalGeometryPointRadius, s.getCircleResolution());
            // pop geometry point matrix
            GLHelper::popMatrix();
        }
        if (endPos != INVALID_DOUBLE) {
            // push geometry point matrix
            GLHelper::pushMatrix();
            glTranslated(myDemandElementGeometry.getShape().back().x(), myDemandElementGeometry.getShape().back().y(), 0.1);
            // draw geometry point
            GLHelper::drawFilledCircle(s.neteditSizeSettings.additionalGeometryPointRadius, s.getCircleResolution());
            // pop geometry point matrix
            GLHelper::popMatrix();
        }
        // pop draw matrix
        GLHelper::popMatrix();
    }
}

/****************************************************************************/
