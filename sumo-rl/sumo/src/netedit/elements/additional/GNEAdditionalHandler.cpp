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
/// @file    GNEAdditionalHandler.cpp
/// @author  Pablo Alvarez Lopez
/// @date    Nov 2015
///
// Builds trigger objects for netedit
/****************************************************************************/
#include <config.h>
#include <netedit/changes/GNEChange_Additional.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNEUndoList.h>
#include <netedit/GNENet.h>

#include "GNEAdditionalHandler.h"
#include "GNEAccess.h"
#include "GNEBusStop.h"
#include "GNECalibrator.h"
#include "GNECalibratorFlow.h"
#include "GNEChargingStation.h"
#include "GNEClosingLaneReroute.h"
#include "GNEClosingReroute.h"
#include "GNEContainerStop.h"
#include "GNEDestProbReroute.h"
#include "GNEDetectorE1.h"
#include "GNEDetectorE1Instant.h"
#include "GNEDetectorE2.h"
#include "GNEDetectorE3.h"
#include "GNEDetectorEntryExit.h"
#include "GNEOverheadWire.h"
#include "GNEPOI.h"
#include "GNEParkingArea.h"
#include "GNEParkingAreaReroute.h"
#include "GNEParkingSpace.h"
#include "GNEPoly.h"
#include "GNERerouter.h"
#include "GNERerouterInterval.h"
#include "GNERerouterSymbol.h"
#include "GNERouteProbReroute.h"
#include "GNERouteProbe.h"
#include "GNETAZ.h"
#include "GNETAZSourceSink.h"
#include "GNETractionSubstation.h"
#include "GNEVaporizer.h"
#include "GNEVariableSpeedSign.h"
#include "GNEVariableSpeedSignStep.h"
#include "GNEVariableSpeedSignSymbol.h"


// ===========================================================================
// GNEAdditionalHandler method definitions
// ===========================================================================

GNEAdditionalHandler::GNEAdditionalHandler(GNENet* net, const bool allowUndoRedo) :
    myNet(net),
    myAllowUndoRedo(allowUndoRedo) {
}


GNEAdditionalHandler::~GNEAdditionalHandler() {
}


void
GNEAdditionalHandler::buildBusStop(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id,
                                   const std::string& laneID, const double startPos, const double endPos, const std::string& name,
                                   const std::vector<std::string>& lines, const int personCapacity, const double parkingLength,
                                   const RGBColor& color, const bool friendlyPosition, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_BUS_STOP, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_BUS_STOP, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_BUS_STOP, SUMO_TAG_LANE);
        } else if (!checkLaneDoublePosition(startPos, endPos, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPosition)) {
            writeErrorInvalidPosition(SUMO_TAG_BUS_STOP, id);
        } else if (personCapacity < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_BUS_STOP, id, SUMO_ATTR_PERSON_CAPACITY);
        } else if (parkingLength < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_BUS_STOP, id, SUMO_ATTR_PARKING_LENGTH);
        } else {
            // build busStop
            GNEAdditional* busStop = new GNEBusStop(SUMO_TAG_BUS_STOP, id, lane, myNet, startPos, endPos, name, lines, personCapacity,
                                                    parkingLength, color, friendlyPosition, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::BUSSTOP, "add " + toString(SUMO_TAG_BUS_STOP) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(busStop, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(busStop);
                lane->addChildElement(busStop);
                busStop->incRef("buildBusStop");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_BUS_STOP, id);
    }
}


void
GNEAdditionalHandler::buildTrainStop(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id,
                                     const std::string& laneID, const double startPos, const double endPos, const std::string& name,
                                     const std::vector<std::string>& lines, const int personCapacity, const double parkingLength,
                                     const RGBColor& color, const bool friendlyPosition, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_TRAIN_STOP, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TRAIN_STOP, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_TRAIN_STOP, SUMO_TAG_LANE);
        } else if (!checkLaneDoublePosition(startPos, endPos, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPosition)) {
            writeErrorInvalidPosition(SUMO_TAG_TRAIN_STOP, id);
        } else if (personCapacity < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_TRAIN_STOP, id, SUMO_ATTR_PERSON_CAPACITY);
        } else if (parkingLength < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_TRAIN_STOP, id, SUMO_ATTR_PARKING_LENGTH);
        } else {
            // build trainStop
            GNEAdditional* trainStop = new GNEBusStop(SUMO_TAG_TRAIN_STOP, id, lane, myNet, startPos, endPos, name, lines, personCapacity,
                    parkingLength, color, friendlyPosition, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TRAINSTOP, "add " + toString(SUMO_TAG_TRAIN_STOP) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(trainStop, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(trainStop);
                lane->addChildElement(trainStop);
                trainStop->incRef("buildTrainStop");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_TRAIN_STOP, id);
    }
}


void
GNEAdditionalHandler::buildAccess(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& laneID,
                                  const double pos, const double length, const bool friendlyPos, const Parameterised::Map& parameters) {
    // get NETEDIT parameters
    NeteditParameters neteditParameters(sumoBaseObject);
    // get lane
    GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
    // get busStop (or trainStop)
    GNEAdditional* busStop = getAdditionalParent(sumoBaseObject, SUMO_TAG_BUS_STOP);
    if (busStop == nullptr) {
        busStop = getAdditionalParent(sumoBaseObject, SUMO_TAG_TRAIN_STOP);
    }
    // Check if busStop parent and lane is correct
    if (lane == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_BUS_STOP, SUMO_TAG_LANE);
    } else if (busStop == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_ACCESS, SUMO_TAG_BUS_STOP);
    } else if (!checkLanePosition(pos, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
        writeErrorInvalidPosition(SUMO_TAG_ACCESS, busStop->getID());
    } else if ((length != -1) && (length < 0)) {
        writeErrorInvalidNegativeValue(SUMO_TAG_ACCESS, busStop->getID(), SUMO_ATTR_LENGTH);
    } else if (!accessCanBeCreated(busStop, lane->getParentEdge())) {
        WRITE_WARNING("Could not build " + toString(SUMO_TAG_ACCESS) + " in netedit; " +  toString(SUMO_TAG_BUS_STOP) + " parent already owns an " + toString(SUMO_TAG_ACCESS) + " in the edge '" + lane->getParentEdge()->getID() + "'");
    } else if (!lane->allowPedestrians()) {
        WRITE_WARNING("Could not build " + toString(SUMO_TAG_ACCESS) + " in netedit; The " + toString(SUMO_TAG_LANE) + " '" + lane->getID() + "' doesn't support pedestrians");
    } else {
        // build access
        GNEAdditional* access = new GNEAccess(busStop, lane, myNet, pos, length, friendlyPos, parameters);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::ACCESS, "add " + toString(SUMO_TAG_ACCESS) + "in  '" + busStop->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(access, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(access);
            lane->addChildElement(access);
            busStop->addChildElement(access);
            access->incRef("buildAccess");
        }
    }
}


void
GNEAdditionalHandler::buildContainerStop(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID,
        const double startPos, const double endPos, const std::string& name, const std::vector<std::string>& lines, const int containerCapacity,
        const double parkingLength, const RGBColor& color, const bool friendlyPosition, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_CONTAINER_STOP, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CONTAINER_STOP, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_CONTAINER_STOP, SUMO_TAG_LANE);
        } else if (!checkLaneDoublePosition(startPos, endPos, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPosition)) {
            writeErrorInvalidPosition(SUMO_TAG_CONTAINER_STOP, id);
        } else if (containerCapacity < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CONTAINER_STOP, id, SUMO_ATTR_CONTAINER_CAPACITY);
        } else if (parkingLength < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CONTAINER_STOP, id, SUMO_ATTR_PARKING_LENGTH);
        } else {
            // build containerStop
            GNEAdditional* containerStop = new GNEContainerStop(id, lane, myNet, startPos, endPos, name, lines, containerCapacity, parkingLength,
                    color, friendlyPosition, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::CONTAINERSTOP, "add " + toString(SUMO_TAG_CONTAINER_STOP) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(containerStop, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(containerStop);
                lane->addChildElement(containerStop);
                containerStop->incRef("buildContainerStop");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_CONTAINER_STOP, id);
    }
}


void
GNEAdditionalHandler::buildChargingStation(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id,
        const std::string& laneID, const double startPos, const double endPos, const std::string& name, const double chargingPower,
        const double efficiency, const bool chargeInTransit, const SUMOTime chargeDelay, const bool friendlyPosition,
        const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_CHARGING_STATION, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CHARGING_STATION, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_CHARGING_STATION, SUMO_TAG_LANE);
        } else if (!checkLaneDoublePosition(startPos, endPos, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPosition)) {
            writeErrorInvalidPosition(SUMO_TAG_CHARGING_STATION, id);
        } else if (chargingPower < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CHARGING_STATION, id, SUMO_ATTR_CHARGINGPOWER);
        } else if (chargeDelay < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CHARGING_STATION, id, SUMO_ATTR_CHARGEDELAY);
        } else {
            // build chargingStation
            GNEAdditional* chargingStation = new GNEChargingStation(id, lane, myNet, startPos, endPos, name, chargingPower, efficiency, chargeInTransit,
                    chargeDelay, friendlyPosition, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::CHARGINGSTATION, "add " + toString(SUMO_TAG_CHARGING_STATION) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(chargingStation, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(chargingStation);
                lane->addChildElement(chargingStation);
                chargingStation->incRef("buildChargingStation");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_CHARGING_STATION, id);
    }

}


void
GNEAdditionalHandler::buildParkingArea(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID,
                                       const double startPos, const double endPos, const std::string& departPos, const std::string& name, const bool friendlyPosition,
                                       const int roadSideCapacity, const bool onRoad, const double width, const double length, const double angle, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_PARKING_AREA, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_PARKING_AREA, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // get departPos double
        const double departPosDouble = GNEAttributeCarrier::canParse<double>(departPos) ? GNEAttributeCarrier::parse<double>(departPos) : 0;
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_PARKING_AREA, SUMO_TAG_LANE);
        } else if (!checkLaneDoublePosition(startPos, endPos, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPosition)) {
            writeErrorInvalidPosition(SUMO_TAG_PARKING_AREA, id);
        } else if (roadSideCapacity < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_PARKING_AREA, id, SUMO_ATTR_ROADSIDE_CAPACITY);
        } else if (width < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_PARKING_AREA, id, SUMO_ATTR_WIDTH);
        } else if (length < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_PARKING_AREA, id, SUMO_ATTR_LENGTH);
        } else if ((departPosDouble < 0) || (departPosDouble > lane->getParentEdge()->getNBEdge()->getFinalLength())) {
            WRITE_ERROR("Could not build " + toString(SUMO_TAG_PARKING_AREA) + " with ID '" + id + "' in netedit; Invalid " + toString(SUMO_ATTR_DEPARTPOS) + " over lane.");
        } else {
            // build parkingArea
            GNEAdditional* parkingArea = new GNEParkingArea(id, lane, myNet, startPos, endPos, GNEAttributeCarrier::canParse<double>(departPos) ? departPos : "",
                    name, friendlyPosition, roadSideCapacity, onRoad,
                    (width == 0) ? SUMO_const_laneWidth : width, length, angle, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::PARKINGAREA, "add " + toString(SUMO_TAG_PARKING_AREA) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(parkingArea, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(parkingArea);
                lane->addChildElement(parkingArea);
                parkingArea->incRef("buildParkingArea");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_PARKING_AREA, id);
    }
}


void
GNEAdditionalHandler::buildParkingSpace(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const double x, const double y, const double z,
                                        const std::string& name, const std::string& width, const std::string& length, const std::string& angle, const double slope,
                                        const Parameterised::Map& parameters) {
    // check width and heights
    if (!width.empty() && !GNEAttributeCarrier::canParse<double>(width)) {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_PARKING_SPACE) + "' in netedit; attribute " +  toString(SUMO_ATTR_WIDTH) + " cannot be parse to float.");
    } else if (!length.empty() && !GNEAttributeCarrier::canParse<double>(length)) {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_PARKING_SPACE) + "' in netedit; attribute " +  toString(SUMO_ATTR_LENGTH) + " cannot be parse to float.");
    } else if (!angle.empty() && !GNEAttributeCarrier::canParse<double>(angle)) {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_PARKING_SPACE) + "' in netedit; attribute " +  toString(SUMO_ATTR_ANGLE) + " cannot be parse to float.");
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNEAdditional* parkingArea = getAdditionalParent(sumoBaseObject, SUMO_TAG_PARKING_AREA);
        // get double values
        const double widthDouble = width.empty() ? 0 : GNEAttributeCarrier::parse<double>(width);
        const double lengthDouble = length.empty() ? 0 : GNEAttributeCarrier::parse<double>(length);
        // check lane
        if (parkingArea == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_PARKING_SPACE, SUMO_TAG_PARKING_AREA);
        } else if (widthDouble < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_PARKING_SPACE, parkingArea->getID(), SUMO_ATTR_WIDTH);
        } else if (lengthDouble < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_PARKING_SPACE, parkingArea->getID(), SUMO_ATTR_LENGTH);
        } else {
            // build parkingSpace
            GNEAdditional* parkingSpace = new GNEParkingSpace(myNet, parkingArea, Position(x, y, z), width, length, angle, slope, name, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::PARKINGSPACE, "add " + toString(SUMO_TAG_PARKING_SPACE) + " in '" + parkingArea->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(parkingSpace, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(parkingSpace);
                parkingArea->addChildElement(parkingSpace);
                parkingSpace->incRef("buildParkingSpace");
            }
            // update geometry (due boundaries)
            parkingSpace->updateGeometry();
        }
    }
}


void
GNEAdditionalHandler::buildE1Detector(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID,
                                      const double position, const SUMOTime frequency, const std::string& file, const std::vector<std::string>& vehicleTypes, const std::string& name,
                                      const bool friendlyPos, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidDetectorID(id)) {
        writeInvalidID(SUMO_TAG_E1DETECTOR, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_E1DETECTOR, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_E1DETECTOR, SUMO_TAG_LANE);
        } else if (!checkLanePosition(position, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
            writeErrorInvalidPosition(SUMO_TAG_E1DETECTOR, id);
        } else if (frequency < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E1DETECTOR, id, SUMO_ATTR_PERIOD);
        } else if (!SUMOXMLDefinitions::isValidFilename(file)) {
            writeErrorInvalidFilename(SUMO_TAG_E1DETECTOR, id);
        } else if (!vehicleTypes.empty() && !SUMOXMLDefinitions::isValidListOfTypeID(vehicleTypes)) {
            writeErrorInvalidVTypes(SUMO_TAG_E1DETECTOR, id);
        } else {
            // build E1
            GNEAdditional* detectorE1 = new GNEDetectorE1(id, lane, myNet, position, frequency, file, vehicleTypes, name, friendlyPos, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::E1, "add " + toString(SUMO_TAG_E1DETECTOR) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(detectorE1, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(detectorE1);
                lane->addChildElement(detectorE1);
                detectorE1->incRef("buildDetectorE1");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_E1DETECTOR, id);
    }
}


void
GNEAdditionalHandler::buildSingleLaneDetectorE2(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID,
        const double pos, const double length, const SUMOTime freq, const std::string& trafficLight, const std::string& filename, const std::vector<std::string>& vehicleTypes,
        const std::string& name, const SUMOTime timeThreshold, const double speedThreshold, const double jamThreshold, const bool friendlyPos,
        const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidDetectorID(id)) {
        writeInvalidID(SUMO_TAG_E2DETECTOR, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_E2DETECTOR, id, false) == nullptr) &&
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_E2DETECTOR_MULTILANE, id, false) == nullptr)) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_E2DETECTOR, SUMO_TAG_LANE);
        } else if (!checkLanePosition(pos, length, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
            writeErrorInvalidPosition(SUMO_TAG_E2DETECTOR, id);
        } else if (length < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_LENGTH);
        } else if ((freq != -1) && (freq < 0)) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_PERIOD);
        } else if ((trafficLight.size() > 0) && !(SUMOXMLDefinitions::isValidNetID(trafficLight))) {
            // temporal
            WRITE_ERROR("Could not build " + toString(SUMO_TAG_E2DETECTOR) + " with ID '" + id + "' in netedit; invalid traffic light ID.");
        } else if (timeThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_TIME_THRESHOLD);
        } else if (speedThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_SPEED_THRESHOLD);
        } else if (jamThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_JAM_DIST_THRESHOLD);
        } else if (timeThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_TIME_THRESHOLD);
        } else if (speedThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_SPEED_THRESHOLD);
        } else if (jamThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_JAM_DIST_THRESHOLD);
        } else if (!SUMOXMLDefinitions::isValidFilename(filename)) {
            writeErrorInvalidFilename(SUMO_TAG_E2DETECTOR, id);
        } else if (!vehicleTypes.empty() && !SUMOXMLDefinitions::isValidListOfTypeID(vehicleTypes)) {
            writeErrorInvalidVTypes(SUMO_TAG_E2DETECTOR, id);
        } else {
            // build E2 single lane
            GNEAdditional* detectorE2 = new GNEDetectorE2(
                id, lane, myNet, pos, length, freq, trafficLight, filename,
                vehicleTypes, name, timeThreshold, speedThreshold, jamThreshold,
                friendlyPos, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::E2, "add " + toString(SUMO_TAG_E2DETECTOR) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(detectorE2, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(detectorE2);
                lane->addChildElement(detectorE2);
                detectorE2->incRef("buildDetectorE2");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_E2DETECTOR, id);
    }
}


void
GNEAdditionalHandler::buildMultiLaneDetectorE2(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::vector<std::string>& laneIDs,
        const double pos, const double endPos, const SUMOTime freq, const std::string& trafficLight, const std::string& filename, const std::vector<std::string>& vehicleTypes,
        const std::string& name, const SUMOTime timeThreshold, const double speedThreshold, const double jamThreshold, const bool friendlyPos,
        const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidDetectorID(id)) {
        writeInvalidID(SUMO_TAG_E2DETECTOR, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_E2DETECTOR, id, false) == nullptr) &&
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_E2DETECTOR_MULTILANE, id, false) == nullptr)) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lanes
        const auto lanes = parseLanes(SUMO_TAG_E2DETECTOR, laneIDs);
        // check lanes
        if (lanes.size() > 0) {
            // calculate path
            if (!GNEAdditional::areLaneConsecutives(lanes)) {
                WRITE_ERROR("Could not build " + toString(SUMO_TAG_E2DETECTOR) + " with ID '" + id + "' in netedit; Lanes aren't consecutives.");
            } else if (!checkMultiLanePosition(
                           pos, lanes.front()->getParentEdge()->getNBEdge()->getFinalLength(),
                           endPos, lanes.back()->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
                writeErrorInvalidPosition(SUMO_TAG_E2DETECTOR, id);
            } else if ((freq != -1) && (freq < 0)) {
                writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_PERIOD);
            } else if ((trafficLight.size() > 0) && !(SUMOXMLDefinitions::isValidNetID(trafficLight))) {
                // temporal
                WRITE_ERROR("Could not build " + toString(SUMO_TAG_E2DETECTOR) + " with ID '" + id + "' in netedit; invalid traffic light ID.");
            } else if (timeThreshold < 0) {
                writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_TIME_THRESHOLD);
            } else if (speedThreshold < 0) {
                writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_HALTING_SPEED_THRESHOLD);
            } else if (jamThreshold < 0) {
                writeErrorInvalidNegativeValue(SUMO_TAG_E2DETECTOR, id, SUMO_ATTR_JAM_DIST_THRESHOLD);
            } else if (!SUMOXMLDefinitions::isValidFilename(filename)) {
                writeErrorInvalidFilename(SUMO_TAG_E2DETECTOR, id);
            } else if (!vehicleTypes.empty() && !SUMOXMLDefinitions::isValidListOfTypeID(vehicleTypes)) {
                writeErrorInvalidVTypes(SUMO_TAG_E2DETECTOR, id);
            } else {
                // build E2 multilane detector
                GNEAdditional* detectorE2 = new GNEDetectorE2(
                    id, lanes, myNet, pos, endPos, freq, trafficLight, filename,
                    vehicleTypes, name, timeThreshold, speedThreshold, jamThreshold,
                    friendlyPos, parameters);
                // insert depending of allowUndoRedo
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::E2, "add " + toString(GNE_TAG_E2DETECTOR_MULTILANE) + " '" + id + "'");
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(detectorE2, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    myNet->getAttributeCarriers()->insertAdditional(detectorE2);
                    for (const auto& lane : lanes) {
                        lane->addChildElement(detectorE2);
                    }
                    detectorE2->incRef("buildDetectorE2Multilane");
                }
            }
        } else {
            writeErrorInvalidLanes(SUMO_TAG_E2DETECTOR, id);
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_E2DETECTOR, id);
    }
}


void
GNEAdditionalHandler::buildDetectorE3(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const Position& pos, const SUMOTime freq,
                                      const std::string& filename, const std::vector<std::string>& vehicleTypes, const std::string& name, const SUMOTime timeThreshold, const double speedThreshold,
                                      const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidDetectorID(id)) {
        writeInvalidID(SUMO_TAG_E3DETECTOR, id);
    } else if (freq < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_E3DETECTOR, id, SUMO_ATTR_PERIOD);
    } else if (timeThreshold < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_E3DETECTOR, id, SUMO_ATTR_HALTING_TIME_THRESHOLD);
    } else if (speedThreshold < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_E3DETECTOR, id, SUMO_ATTR_HALTING_SPEED_THRESHOLD);
    } else if (!SUMOXMLDefinitions::isValidFilename(filename)) {
        writeErrorInvalidFilename(SUMO_TAG_E3DETECTOR, id);
    } else if (!vehicleTypes.empty() && !SUMOXMLDefinitions::isValidListOfTypeID(vehicleTypes)) {
        writeErrorInvalidVTypes(SUMO_TAG_E3DETECTOR, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_E3DETECTOR, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // build E3
        GNEAdditional* E3 = new GNEDetectorE3(id, myNet, pos, freq, filename, vehicleTypes, name, timeThreshold, speedThreshold, parameters);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::E3, "add " + toString(SUMO_TAG_E3DETECTOR) + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(E3, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(E3);
            E3->incRef("buildDetectorE3");
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_E3DETECTOR, id);
    }
}


void
GNEAdditionalHandler::buildDetectorEntry(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& laneID, const double pos,
        const bool friendlyPos, const Parameterised::Map& parameters) {
    // get lane
    GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
    // get E3 parent
    GNEAdditional* E3 = getAdditionalParent(sumoBaseObject, SUMO_TAG_E3DETECTOR);
    // Check if Detector E3 parent and lane is correct
    if (lane == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_ENTRY, SUMO_TAG_LANE);
    } else if (E3 == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_ENTRY, SUMO_TAG_E3DETECTOR);
    } else if (!checkLanePosition(pos, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
        writeErrorInvalidPosition(SUMO_TAG_ENTRY, E3->getID());
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // build entry instant
        GNEAdditional* entry = new GNEDetectorEntryExit(SUMO_TAG_DET_ENTRY, myNet, E3, lane, pos, friendlyPos, parameters);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::E3ENTRY, "add " + toString(SUMO_TAG_DET_ENTRY) + " in '" + E3->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(entry, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(entry);
            lane->addChildElement(entry);
            E3->addChildElement(entry);
            entry->incRef("buildDetectorEntry");
        }
    }
}


void
GNEAdditionalHandler::buildDetectorExit(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& laneID, const double pos,
                                        const bool friendlyPos, const Parameterised::Map& parameters) {
    // get lane
    GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
    // get E3 parent
    GNEAdditional* E3 = getAdditionalParent(sumoBaseObject, SUMO_TAG_E3DETECTOR);
    // Check if Detector E3 parent and lane is correct
    if (lane == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_DET_EXIT, SUMO_TAG_LANE);
    } else if (E3 == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_DET_EXIT, SUMO_TAG_E3DETECTOR);
    } else if (!checkLanePosition(pos, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
        writeErrorInvalidPosition(SUMO_TAG_ENTRY, E3->getID());
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // build exit instant
        GNEAdditional* exit = new GNEDetectorEntryExit(SUMO_TAG_DET_EXIT, myNet, E3, lane, pos, friendlyPos, parameters);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::E3EXIT, "add " + toString(SUMO_TAG_DET_EXIT) + " in '" + E3->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(exit, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(exit);
            lane->addChildElement(exit);
            E3->addChildElement(exit);
            exit->incRef("buildDetectorExit");
        }
    }
}


void
GNEAdditionalHandler::buildDetectorE1Instant(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID, double pos,
        const std::string& filename, const std::vector<std::string>& vehicleTypes, const std::string& name, const bool friendlyPos, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidDetectorID(id)) {
        writeInvalidID(SUMO_TAG_INSTANT_INDUCTION_LOOP, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_INSTANT_INDUCTION_LOOP, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_INDUCTION_LOOP, SUMO_TAG_LANE);
        } else if (!SUMOXMLDefinitions::isValidFilename(filename)) {
            writeErrorInvalidFilename(SUMO_TAG_INSTANT_INDUCTION_LOOP, id);
        } else if (!checkLanePosition(pos, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
            writeErrorInvalidPosition(SUMO_TAG_INSTANT_INDUCTION_LOOP, id);
        } else {
            // build E1 instant
            GNEAdditional* detectorE1Instant = new GNEDetectorE1Instant(id, lane, myNet, pos, filename, vehicleTypes, name, friendlyPos, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::E1INSTANT, "add " + toString(SUMO_TAG_INSTANT_INDUCTION_LOOP) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(detectorE1Instant, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(detectorE1Instant);
                lane->addChildElement(detectorE1Instant);
                detectorE1Instant->incRef("buildDetectorE1Instant");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_INSTANT_INDUCTION_LOOP, id);
    }
}


void
GNEAdditionalHandler::buildLaneCalibrator(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& laneID, const double pos,
        const std::string& name, const std::string& outfile, const SUMOTime freq, const std::string& routeprobeID, const double jamThreshold, const std::vector<std::string>& vTypes,
        const Parameterised::Map& parameters) {
    // get lane
    GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
    // get routeProbe
    GNEAdditional* routeProbe = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_ROUTEPROBE, routeprobeID, false);
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_CALIBRATOR, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CALIBRATOR, id, false) != nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_CALIBRATOR_LANE, id, false) != nullptr)) {
        writeErrorDuplicated(SUMO_TAG_CALIBRATOR, id);
    } else if ((routeprobeID.size() > 0) && (routeProbe == nullptr)) {
        writeErrorInvalidParent(SUMO_TAG_CALIBRATOR, SUMO_TAG_ROUTEPROBE);
    } else if (lane == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CALIBRATOR, SUMO_TAG_LANE);
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // check lane
        if (!checkLanePosition(pos, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), false)) {
            writeErrorInvalidPosition(SUMO_TAG_CALIBRATOR, id);
        } else if (freq < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CALIBRATOR, id, SUMO_ATTR_PERIOD);
        } else if (jamThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CALIBRATOR, id, SUMO_ATTR_JAM_DIST_THRESHOLD);
        } else {
            // build Calibrator
            GNEAdditional* calibrator = (routeProbe == nullptr) ?
                                        new GNECalibrator(id, myNet, lane, pos, freq, name, outfile, jamThreshold, vTypes, parameters) :
                                        new GNECalibrator(id, myNet, lane, pos, freq, name, outfile, routeProbe, jamThreshold, vTypes, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::CALIBRATOR, "add " + toString(SUMO_TAG_CALIBRATOR) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(calibrator, true), true);
                myNet->getViewNet()->getUndoList()->end();
                // center after creation
                if (neteditParameters.centerAfterCreation) {
                    myNet->getViewNet()->centerTo(calibrator->getPositionInView(), false);
                }
            } else {
                myNet->getAttributeCarriers()->insertAdditional(calibrator);
                lane->addChildElement(calibrator);
                if (routeProbe) {
                    routeProbe->addChildElement(calibrator);
                }
                calibrator->incRef("buildCalibrator");
            }
        }
    }
}


void
GNEAdditionalHandler::buildEdgeCalibrator(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& edgeID, const double pos,
        const std::string& name, const std::string& outfile, const SUMOTime freq, const std::string& routeprobeID, const double jamThreshold, const std::vector<std::string>& vTypes,
        const Parameterised::Map& parameters) {
    // get edge
    GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
    // get routeProbe
    GNEAdditional* routeProbe = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_ROUTEPROBE, routeprobeID, false);
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_CALIBRATOR, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_CALIBRATOR, id, false) != nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_CALIBRATOR_LANE, id, false) != nullptr)) {
        writeErrorDuplicated(SUMO_TAG_CALIBRATOR, id);
    } else if ((routeprobeID.size() > 0) && (routeProbe == nullptr)) {
        writeErrorInvalidParent(SUMO_TAG_CALIBRATOR, SUMO_TAG_ROUTEPROBE);
    } else if (edge == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CALIBRATOR, SUMO_TAG_EDGE);
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        if (!checkLanePosition(pos, 0, edge->getLanes().front()->getParentEdge()->getNBEdge()->getFinalLength(), false)) {
            writeErrorInvalidPosition(SUMO_TAG_CALIBRATOR, id);
        } else if (freq < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CALIBRATOR, id, SUMO_ATTR_PERIOD);
        } else if (jamThreshold < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_CALIBRATOR, id, SUMO_ATTR_JAM_DIST_THRESHOLD);
        } else {
            // build Calibrator
            GNEAdditional* calibrator = (routeProbe == nullptr) ?
                                        new GNECalibrator(id, myNet, edge, pos, freq, name, outfile, jamThreshold, vTypes, parameters) :
                                        new GNECalibrator(id, myNet, edge, pos, freq, name, outfile, routeProbe, jamThreshold, vTypes, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::CALIBRATOR, "add " + toString(SUMO_TAG_CALIBRATOR) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(calibrator, true), true);
                myNet->getViewNet()->getUndoList()->end();
                // center after creation
                if (neteditParameters.centerAfterCreation) {
                    myNet->getViewNet()->centerTo(calibrator->getPositionInView(), false);
                }
            } else {
                myNet->getAttributeCarriers()->insertAdditional(calibrator);
                edge->addChildElement(calibrator);
                if (routeProbe) {
                    routeProbe->addChildElement(calibrator);
                }
                calibrator->incRef("buildCalibrator");
            }
        }
    }
}


void
GNEAdditionalHandler::buildCalibratorFlow(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const SUMOVehicleParameter& vehicleParameter) {
    // get vType
    GNEDemandElement* vType = myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_VTYPE, vehicleParameter.vtypeid.empty() ? DEFAULT_VTYPE_ID : vehicleParameter.vtypeid, false);
    // get route
    GNEDemandElement* route = myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_ROUTE, vehicleParameter.routeid, false);
    // get calibrator parent
    GNEAdditional* calibrator = myNet->getAttributeCarriers()->retrieveAdditional(sumoBaseObject->getParentSumoBaseObject()->getTag(), sumoBaseObject->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID), false);
    // check parents
    if (vType == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_FLOW, SUMO_TAG_VTYPE);
    } else if (route == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_FLOW, SUMO_TAG_ROUTE);
    } else if (calibrator == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_FLOW, SUMO_TAG_CALIBRATOR);
    } else {
        // create calibrator flow
        GNEAdditional* flow = new GNECalibratorFlow(calibrator, vType, route, vehicleParameter);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::FLOW, "add " + flow->getTagStr() + " in '" + calibrator->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(flow, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            calibrator->addChildElement(flow);
            route->addChildElement(flow);
            vType->addChildElement(flow);
            flow->incRef("buildCalibratorFlow");
        }
    }
}


void
GNEAdditionalHandler::buildRerouter(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const Position& pos,
                                    const std::vector<std::string>& edgeIDs, const double prob, const std::string& name, const bool off, const SUMOTime timeThreshold,
                                    const std::vector<std::string>& vTypes, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_REROUTER, id);
    } else if (prob < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_REROUTER, id, SUMO_ATTR_PROB);
    } else if (timeThreshold < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_REROUTER, id, SUMO_ATTR_HALTING_TIME_THRESHOLD);
    } else if (!vTypes.empty() && !SUMOXMLDefinitions::isValidListOfTypeID(vTypes)) {
        writeErrorInvalidVTypes(SUMO_TAG_REROUTER, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_REROUTER, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // parse edges
        std::vector<GNEEdge*> edges = parseEdges(SUMO_TAG_REROUTER, edgeIDs);
        // check edges
        if (edges.size() > 0) {
            GNEAdditional* rerouter = nullptr;
            // continue depending of position
            if (pos == Position::INVALID) {
                if (edges.size() > 0) {
                    PositionVector laneShape = edges.front()->getLanes().front()->getLaneShape();
                    // move to side
                    laneShape.move2side(3);
                    // create rerouter
                    rerouter = new GNERerouter(id, myNet, laneShape.positionAtOffset2D(laneShape.length2D() - 6), name, prob, off, timeThreshold, vTypes, parameters);
                } else {
                    rerouter = new GNERerouter(id, myNet, Position(0, 0), name, prob, off, timeThreshold, vTypes, parameters);
                }
            } else {
                rerouter = new GNERerouter(id, myNet, pos, name, prob, off, timeThreshold, vTypes, parameters);
            }
            // create rerouter Symbols
            std::vector<GNEAdditional*> rerouterSymbols;
            for (const auto& edge : edges) {
                rerouterSymbols.push_back(new GNERerouterSymbol(rerouter, edge));
            }
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::REROUTER, "add " + toString(SUMO_TAG_REROUTER) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(rerouter, true), true);
                // add symbols
                for (const auto& rerouterSymbol : rerouterSymbols) {
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(rerouterSymbol, true), true);
                }
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(rerouter);
                rerouter->incRef("buildRerouter");
                // add symbols into rerouter
                for (const auto& rerouterSymbol : rerouterSymbols) {
                    rerouter->addChildElement(rerouterSymbol);
                }
                // add symbols into edges
                for (int i = 0; i < (int)edges.size(); i++) {
                    edges.at(i)->addChildElement(rerouterSymbols.at(i));
                }
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_REROUTER, id);
    }
}


void
GNEAdditionalHandler::buildRerouterInterval(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const SUMOTime begin, const SUMOTime end) {
    // get rerouter parent
    GNEAdditional* rerouter = getAdditionalParent(sumoBaseObject, SUMO_TAG_REROUTER);
    // check if rerouter exist
    if (rerouter == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_INTERVAL, SUMO_TAG_REROUTER);
    } else if (begin < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_INTERVAL, rerouter->getID(), SUMO_ATTR_BEGIN);
    } else if (end < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_INTERVAL, rerouter->getID(), SUMO_ATTR_END);
    } else if (end < begin) {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_INTERVAL) + " with ID '" + rerouter->getID() + "' in netedit; " +  toString(SUMO_ATTR_BEGIN) + " is greather than " + toString(SUMO_ATTR_END) + ".");
    } else {
        // check if new interval will produce a overlapping
        if (checkOverlappingRerouterIntervals(rerouter, begin, end)) {
            // create rerouter interval and add it into rerouter parent
            GNEAdditional* rerouterInterval = new GNERerouterInterval(rerouter, begin, end);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::REROUTERINTERVAL, "add " + rerouterInterval->getTagStr() + " in '" + rerouter->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(rerouterInterval, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                rerouter->addChildElement(rerouterInterval);
                rerouterInterval->incRef("buildRerouterInterval");
            }
        } else {
            WRITE_ERROR("Could not build " + toString(SUMO_TAG_INTERVAL) + " with begin '" + toString(begin) + "' and '" + toString(end) + "' in '" + rerouter->getID() + "' due overlapping.");
        }
    }
}


void
GNEAdditionalHandler::buildClosingLaneReroute(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& closedLaneID, SVCPermissions permissions) {
    // get rerouter interval parent
    GNEAdditional* rerouterInterval = getRerouterIntervalParent(sumoBaseObject);
    // get closed lane
    GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(closedLaneID, false);
    // check parents
    if (lane == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CLOSING_LANE_REROUTE, SUMO_TAG_LANE);
    } else if (rerouterInterval == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CLOSING_LANE_REROUTE, SUMO_TAG_INTERVAL);
    } else {
        // create closing lane reroute
        GNEAdditional* closingLaneReroute = new GNEClosingLaneReroute(rerouterInterval, lane, permissions);
        // add it to interval parent depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::CLOSINGLANEREROUTE, "add " + closingLaneReroute->getTagStr() + " in '" + lane->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(closingLaneReroute, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            rerouterInterval->addChildElement(closingLaneReroute);
            closingLaneReroute->incRef("buildClosingLaneReroute");
        }
    }
}


void
GNEAdditionalHandler::buildClosingReroute(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& closedEdgeID, SVCPermissions permissions) {
    // get rerouter interval parent
    GNEAdditional* rerouterInterval = getRerouterIntervalParent(sumoBaseObject);
    // get closed edge
    GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(closedEdgeID, false);
    // check parents
    if (edge == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CLOSING_REROUTE, SUMO_TAG_EDGE);
    } else if (rerouterInterval == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_CLOSING_REROUTE, SUMO_TAG_INTERVAL);
    } else {
        // create closing reroute
        GNEAdditional* closingLaneReroute = new GNEClosingReroute(rerouterInterval, edge, permissions);
        // add it to interval parent depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::CLOSINGREROUTE, "add " + closingLaneReroute->getTagStr() + " in '" + edge->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(closingLaneReroute, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            rerouterInterval->addChildElement(closingLaneReroute);
            closingLaneReroute->incRef("buildClosingLaneReroute");
        }
    }
}

void
GNEAdditionalHandler::buildDestProbReroute(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& newEdgeDestinationID, const double probability) {
    // get rerouter interval parent
    GNEAdditional* rerouterInterval = getRerouterIntervalParent(sumoBaseObject);
    // get edge
    GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(newEdgeDestinationID, false);
    // check parents
    if (edge == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_DEST_PROB_REROUTE, SUMO_TAG_EDGE);
    } else if (rerouterInterval == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_DEST_PROB_REROUTE, SUMO_TAG_INTERVAL);
    } else {
        // create dest probability reroute
        GNEAdditional* destProbReroute = new GNEDestProbReroute(rerouterInterval, edge, probability);
        // add it to interval parent depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::DESTPROBREROUTE, "add " + destProbReroute->getTagStr() + " in '" + edge->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(destProbReroute, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            rerouterInterval->addChildElement(destProbReroute);
            destProbReroute->incRef("builDestProbReroute");
        }
    }
}


void
GNEAdditionalHandler::buildParkingAreaReroute(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& newParkignAreaID, const double probability, const bool visible) {
    // get rerouter interval parent
    GNEAdditional* rerouterInterval = getRerouterIntervalParent(sumoBaseObject);
    // get parking area
    GNEAdditional* parkingArea = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_PARKING_AREA, newParkignAreaID, false);
    // check parents
    if (rerouterInterval == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_PARKING_AREA_REROUTE, SUMO_TAG_INTERVAL);
    } else if (parkingArea == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_PARKING_AREA_REROUTE, SUMO_TAG_PARKING_AREA);
    } else {
        // create parking area reroute
        GNEAdditional* parkingAreaReroute = new GNEParkingAreaReroute(rerouterInterval, parkingArea, probability, visible);
        // add it to interval parent depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::PARKINGZONEREROUTE, "add " + parkingAreaReroute->getTagStr() + " in '" + parkingArea->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(parkingAreaReroute, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            rerouterInterval->addChildElement(parkingAreaReroute);
            parkingAreaReroute->incRef("builParkingAreaReroute");
        }
    }
}


void
GNEAdditionalHandler::buildRouteProbReroute(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& newRouteID, const double probability) {
    // get rerouter interval parent
    GNEAdditional* rerouterInterval = getRerouterIntervalParent(sumoBaseObject);
    // get route parent
    GNEDemandElement* route = myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_ROUTE, newRouteID, false);
    // check parents
    if (rerouterInterval == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_ROUTE_PROB_REROUTE, SUMO_TAG_INTERVAL);
    } else if (route == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_ROUTE_PROB_REROUTE, SUMO_TAG_ROUTE);
    } else {
        // create rout prob reroute
        GNEAdditional* routeProbReroute = new GNERouteProbReroute(rerouterInterval, route, probability);
        // add it to interval parent depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::ROUTEPROBREROUTE, "add " + routeProbReroute->getTagStr() + " in '" + route->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(routeProbReroute, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            rerouterInterval->addChildElement(routeProbReroute);
            routeProbReroute->incRef("buildRouteProbReroute");
        }
    }
}


void
GNEAdditionalHandler::buildRouteProbe(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& edgeID, const SUMOTime freq,
                                      const std::string& name, const std::string& file, const SUMOTime begin, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_ROUTEPROBE, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_ROUTEPROBE, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get edge
        GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
        // check lane
        if (edge == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_ROUTEPROBE, SUMO_TAG_EDGE);
        } else if (freq < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_ROUTEPROBE, id, SUMO_ATTR_PERIOD);
        } else if (begin < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_ROUTEPROBE, id, SUMO_ATTR_BEGIN);
        } else if (!SUMOXMLDefinitions::isValidFilename(file)) {
            writeErrorInvalidFilename(SUMO_TAG_ROUTEPROBE, id);
        } else {
            // build route probe
            GNEAdditional* routeProbe = new GNERouteProbe(id, myNet, edge, freq, name, file, begin, parameters);
            // insert depending of allowUndoRedo
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::ROUTEPROBE, "add " + toString(SUMO_TAG_ROUTEPROBE) + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(routeProbe, true), true);
                myNet->getViewNet()->getUndoList()->end();
                // center after creation
                if (neteditParameters.centerAfterCreation) {
                    myNet->getViewNet()->centerTo(routeProbe->getPositionInView(), false);
                }
            } else {
                myNet->getAttributeCarriers()->insertAdditional(routeProbe);
                edge->addChildElement(routeProbe);
                routeProbe->incRef("buildRouteProbe");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_ROUTEPROBE, id);
    }
}


void
GNEAdditionalHandler::buildVariableSpeedSign(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const Position& pos,
        const std::vector<std::string>& laneIDs, const std::string& name, const std::vector<std::string>& vTypes, const Parameterised::Map& parameters) {
    /// check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_VSS, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_VSS, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // parse lanes
        std::vector<GNELane*> lanes = parseLanes(SUMO_TAG_VSS, laneIDs);
        // check lane
        if (lanes.size() > 0) {
            // check vTypes
            if (!vTypes.empty() && !checkListOfVehicleTypes(vTypes)) {
                writeErrorInvalidVTypes(SUMO_TAG_VSS, id);
            } else {
                // create VSS
                GNEAdditional* variableSpeedSign = new GNEVariableSpeedSign(id, myNet, pos, name, vTypes, parameters);
                // create VSS Symbols
                std::vector<GNEAdditional*> VSSSymbols;
                for (const auto& lane : lanes) {
                    VSSSymbols.push_back(new GNEVariableSpeedSignSymbol(variableSpeedSign, lane));
                }
                // insert depending of allowUndoRedo
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::VARIABLESPEEDSIGN, "add " + toString(SUMO_TAG_VSS) + " '" + id + "'");
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(variableSpeedSign, true), true);
                    for (const auto& VSSSymbol : VSSSymbols) {
                        myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(VSSSymbol, true), true);
                    }
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    myNet->getAttributeCarriers()->insertAdditional(variableSpeedSign);
                    variableSpeedSign->incRef("buildVariableSpeedSign");
                    // add symbols into VSS
                    for (const auto& VSSSymbol : VSSSymbols) {
                        variableSpeedSign->addChildElement(VSSSymbol);
                    }
                    // add symbols into lanes
                    for (int i = 0; i < (int)lanes.size(); i++) {
                        lanes.at(i)->addChildElement(VSSSymbols.at(i));
                    }
                }
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_VSS, id);
    }
}


void
GNEAdditionalHandler::buildVariableSpeedSignStep(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const SUMOTime time, const std::string& speed) {
    // get VSS parent
    GNEAdditional* VSS = getAdditionalParent(sumoBaseObject, SUMO_TAG_VSS);
    // check lane
    if (VSS == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_STEP, SUMO_TAG_VSS);
    } else if (time < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_STEP, VSS->getID(), SUMO_ATTR_BEGIN);
        /*
            } else if (speed < 0) {
                writeErrorInvalidNegativeValue(SUMO_TAG_STEP, VSS->getID(), SUMO_ATTR_SPEED);
        */
    } else {
        // create Variable Speed Sign
        GNEAdditional* variableSpeedSignStep = new GNEVariableSpeedSignStep(VSS, time, speed);
        // add it depending of allow undoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::VSSSTEP, "add " + variableSpeedSignStep->getTagStr() + " in '" + VSS->getID() + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(variableSpeedSignStep, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            VSS->addChildElement(variableSpeedSignStep);
            variableSpeedSignStep->incRef("buildVariableSpeedSignStep");
        }
    }
}


void
GNEAdditionalHandler::buildVaporizer(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& edgeID, const SUMOTime beginTime,
                                     const SUMOTime endTime, const std::string& name, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(edgeID)) {
        writeInvalidID(SUMO_TAG_VAPORIZER, edgeID);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_VAPORIZER, edgeID, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get edge
        GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
        // check lane
        if (edge == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_VAPORIZER, SUMO_TAG_EDGE);
        } else if (beginTime < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_VAPORIZER, edge->getID(), SUMO_ATTR_BEGIN);
        } else if (endTime < 0) {
            writeErrorInvalidNegativeValue(SUMO_TAG_VAPORIZER, edge->getID(), SUMO_ATTR_END);
        } else if (endTime < beginTime) {
            WRITE_ERROR("Could not build " + toString(SUMO_TAG_VAPORIZER) + " with ID '" + edge->getID() + "' in netedit; " +  toString(SUMO_ATTR_BEGIN) + " is greather than " + toString(SUMO_ATTR_END) + ".");
        } else {
            // build vaporizer
            GNEAdditional* vaporizer = new GNEVaporizer(myNet, edge, beginTime, endTime, name, parameters);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::VAPORIZER, "add " + toString(SUMO_TAG_VAPORIZER) + " in '" + edge->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(vaporizer, true), true);
                myNet->getViewNet()->getUndoList()->end();
                // center after creation
                if (neteditParameters.centerAfterCreation) {
                    myNet->getViewNet()->centerTo(vaporizer->getPositionInView(), false);
                }
            } else {
                myNet->getAttributeCarriers()->insertAdditional(vaporizer);
                edge->addChildElement(vaporizer);
                vaporizer->incRef("buildVaporizer");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_VAPORIZER, edgeID);
    }
}


void
GNEAdditionalHandler::buildTAZ(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const PositionVector& shape,
                               const Position& center, const bool fill, const RGBColor& color, const std::vector<std::string>& edgeIDs,
                               const std::string& name, const Parameterised::Map& parameters) {
    // parse edges
    const std::vector<GNEEdge*> edges = parseEdges(SUMO_TAG_TAZ, edgeIDs);
    // check TAZShape
    PositionVector TAZShape = shape;
    if (TAZShape.size() == 0) {
        // declare boundary
        Boundary TAZBoundary;
        for (const auto& edge : edges) {
            TAZBoundary.add(edge->getCenteringBoundary());
        }
        // iterate over children and add sourceSinkEdge boundaries
        for (const auto& sourceSink : sumoBaseObject->getSumoBaseObjectChildren()) {
            const GNEEdge* sourceSinkEdge = myNet->getAttributeCarriers()->retrieveEdge(sourceSink->getStringAttribute(SUMO_ATTR_ID), false);
            if (sourceSinkEdge) {
                TAZBoundary.add(sourceSinkEdge->getCenteringBoundary());
            }
        }
        // update TAZShape
        TAZShape = TAZBoundary.getShape(true);
    }
    // check TAZ
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_TAZ, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TAZ, id, false) != nullptr) {
        writeErrorDuplicated(SUMO_TAG_TAZ, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_POLY, id, false) != nullptr) {
        writeErrorDuplicated(SUMO_TAG_TAZ, id);
    } else if (TAZShape.size() == 0) {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_TAZ) + " with ID '" + id + "' in netedit; Invalid Shape.");
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // build TAZ with the given shape
        GNEAdditional* TAZ = new GNETAZ(id, myNet, TAZShape, center, fill, color, name, parameters);
        // disable updating geometry of TAZ children during insertion (because in large nets provokes slowdowns)
        myNet->disableUpdateGeometry();
        // add it depending of allow undoRed
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "add " + toString(SUMO_TAG_TAZ) + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZ, true), true);
            // create TAZEdges
            for (const auto& edge : edges) {
                // create TAZ Source using GNEChange_Additional
                GNEAdditional* TAZSource = new GNETAZSourceSink(SUMO_TAG_TAZSOURCE, TAZ, edge, 1);
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSource, true), true);
                // create TAZ Sink using GNEChange_Additional
                GNEAdditional* TAZSink = new GNETAZSourceSink(SUMO_TAG_TAZSINK, TAZ, edge, 1);
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSink, true), true);
            }
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(TAZ);
            TAZ->incRef("buildTAZ");
            for (const auto& edge : edges) {
                // create TAZ Source
                GNEAdditional* TAZSource = new GNETAZSourceSink(SUMO_TAG_TAZSOURCE, TAZ, edge, 1);
                TAZSource->incRef("buildTAZ");
                TAZ->addChildElement(TAZSource);
                // create TAZ Sink
                GNEAdditional* TAZSink = new GNETAZSourceSink(SUMO_TAG_TAZSINK, TAZ, edge, 1);
                TAZSink->incRef("buildTAZ");
                TAZ->addChildElement(TAZSink);
            }
        }
        // enable updating geometry again and update geometry of TAZ
        myNet->enableUpdateGeometry();
        // update TAZ parent
        TAZ->updateGeometry();
    }
}


void
GNEAdditionalHandler::buildTAZSource(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& edgeID, const double departWeight) {
    // get TAZ parent
    GNEAdditional* TAZ = getAdditionalParent(sumoBaseObject, SUMO_TAG_TAZ);
    // get edge
    GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
    // declare TAZ Sink
    GNEAdditional* TAZSink = nullptr;
    // check parents
    if (TAZ == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_SOURCE, SUMO_TAG_TAZ);
    } else if (edge == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_SOURCE, SUMO_TAG_EDGE);
    } else {
        // first check if a TAZSink in the same edge for the same TAZ
        for (const auto& TAZElement : TAZ->getChildAdditionals()) {
            if ((TAZElement->getTagProperty().getTag() == SUMO_TAG_TAZSINK) && (TAZElement->getAttribute(SUMO_ATTR_EDGE) == edge->getID())) {
                TAZSink = TAZElement;
            }
        }
        // check if TAZSink has to be created
        if (TAZSink == nullptr) {
            // Create TAZ with weight 0 (default)
            TAZSink = new GNETAZSourceSink(SUMO_TAG_TAZSINK, TAZ, edge, 0);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "add " + toString(SUMO_TAG_TAZSINK) + " in '" + TAZ->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSink, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(TAZSink);
                TAZSink->incRef("buildTAZSource");
            }
        }
        // now check check if TAZSource exist
        GNEAdditional* TAZSource = nullptr;
        // first check if a TAZSink in the same edge for the same TAZ
        for (const auto& TAZElement : TAZ->getChildAdditionals()) {
            if ((TAZElement->getTagProperty().getTag() == SUMO_TAG_TAZSOURCE) && (TAZElement->getAttribute(SUMO_ATTR_EDGE) == edge->getID())) {
                TAZSource = TAZElement;
            }
        }
        // check if TAZSource has to be created
        if (TAZSource == nullptr) {
            // Create TAZ only with departWeight
            TAZSource = new GNETAZSourceSink(SUMO_TAG_TAZSOURCE, TAZ, edge, departWeight);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "add " + toString(SUMO_TAG_TAZSOURCE) + " in '" + TAZ->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSource, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(TAZSource);
                TAZSource->incRef("buildTAZSource");
            }
        } else {
            // update TAZ Attribute depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "update " + toString(SUMO_TAG_TAZSOURCE) + " in '" + TAZ->getID() + "'");
                TAZSource->setAttribute(SUMO_ATTR_WEIGHT, toString(departWeight), myNet->getViewNet()->getUndoList());
                myNet->getViewNet()->getUndoList()->end();
            } else {
                TAZSource->setAttribute(SUMO_ATTR_WEIGHT, toString(departWeight), nullptr);
                TAZSource->incRef("buildTAZSource");
            }
        }
    }
}


void
GNEAdditionalHandler::buildTAZSink(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& edgeID, const double arrivalWeight) {
    // get TAZ parent
    GNEAdditional* TAZ = getAdditionalParent(sumoBaseObject, SUMO_TAG_TAZ);
    // get edge
    GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
    // check parents
    if (TAZ == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_SINK, SUMO_TAG_TAZ);
    } else if (edge == nullptr) {
        writeErrorInvalidParent(SUMO_TAG_SINK, SUMO_TAG_EDGE);
    } else {
        // declare TAZ source
        GNEAdditional* TAZSource = nullptr;
        // first check if a TAZSink in the same edge for the same TAZ
        for (const auto& TAZElement : TAZ->getChildAdditionals()) {
            if ((TAZElement->getTagProperty().getTag() == SUMO_TAG_TAZSOURCE) && (TAZElement->getAttribute(SUMO_ATTR_EDGE) == edge->getID())) {
                TAZSource = TAZElement;
            }
        }
        // check if TAZSource has to be created
        if (TAZSource == nullptr) {
            // Create TAZ with empty value
            TAZSource = new GNETAZSourceSink(SUMO_TAG_TAZSOURCE, TAZ, edge, 0);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "add " + toString(SUMO_TAG_TAZSOURCE) + " in '" + TAZ->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSource, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(TAZSource);
                TAZSource->incRef("buildTAZSink");
            }
        }
        GNEAdditional* TAZSink = nullptr;
        // first check if a TAZSink in the same edge for the same TAZ
        for (const auto& TAZElement : TAZ->getChildAdditionals()) {
            if ((TAZElement->getTagProperty().getTag() == SUMO_TAG_TAZSINK) && (TAZElement->getAttribute(SUMO_ATTR_EDGE) == edge->getID())) {
                TAZSink = TAZElement;
            }
        }
        // check if TAZSink has to be created
        if (TAZSink == nullptr) {
            // Create TAZ only with arrivalWeight
            TAZSink = new GNETAZSourceSink(SUMO_TAG_TAZSINK, TAZ, edge, arrivalWeight);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "add " + toString(SUMO_TAG_TAZSINK) + " in '" + TAZ->getID() + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(TAZSink, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                myNet->getAttributeCarriers()->insertAdditional(TAZSink);
                TAZSink->incRef("buildTAZSink");
            }
        } else {
            // update TAZ Attribute depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZ, "update " + toString(SUMO_TAG_TAZSINK) + " in '" + TAZ->getID() + "'");
                TAZSink->setAttribute(SUMO_ATTR_WEIGHT, toString(arrivalWeight), myNet->getViewNet()->getUndoList());
                myNet->getViewNet()->getUndoList()->end();
            } else {
                TAZSink->setAttribute(SUMO_ATTR_WEIGHT, toString(arrivalWeight), nullptr);
                TAZSink->incRef("buildTAZSink");
            }
        }
    }
}


void
GNEAdditionalHandler::buildTractionSubstation(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const Position& pos,
        const double voltage, const double currentLimit, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_TRACTION_SUBSTATION, id);
    } else if (voltage < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_TRACTION_SUBSTATION, id, SUMO_ATTR_VOLTAGE);
    } else if (currentLimit < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_TRACTION_SUBSTATION, id, SUMO_ATTR_CURRENTLIMIT);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TRACTION_SUBSTATION, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // build traction substation
        GNEAdditional* tractionSubstation = new GNETractionSubstation(id, myNet, pos, voltage, currentLimit, parameters);
        // insert depending of allowUndoRedo
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::TRACTION_SUBSTATION, "add " + toString(SUMO_TAG_TRACTION_SUBSTATION) + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(tractionSubstation, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            myNet->getAttributeCarriers()->insertAdditional(tractionSubstation);
            tractionSubstation->incRef("buildTractionSubstation");
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_TRACTION_SUBSTATION, id);
    }
}


void
GNEAdditionalHandler::buildOverheadWire(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& substationId,
                                        const std::vector<std::string>& laneIDs, const double startPos, const double endPos, const bool friendlyPos,
                                        const std::vector<std::string>& forbiddenInnerLanes, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_OVERHEAD_WIRE_SECTION, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_OVERHEAD_WIRE_SECTION, id, false) == nullptr) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lanes
        const auto lanes = parseLanes(SUMO_TAG_OVERHEAD_WIRE_SECTION, laneIDs);
        // get traction substation
        const auto tractionSubstation = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TRACTION_SUBSTATION, substationId, false);
        // check lanes
        if (lanes.size() > 0) {
            // calculate path
            if (!GNEAdditional::areLaneConsecutives(lanes)) {
                WRITE_ERROR("Could not build " + toString(SUMO_TAG_OVERHEAD_WIRE_SECTION) + " with ID '" + id + "' in netedit; Lanes aren't consecutives.");
            } else if (!checkMultiLanePosition(
                           startPos, lanes.front()->getParentEdge()->getNBEdge()->getFinalLength(),
                           endPos, lanes.back()->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
                writeErrorInvalidPosition(SUMO_TAG_OVERHEAD_WIRE_SECTION, id);
            } else if (tractionSubstation == nullptr) {
                writeErrorInvalidParent(SUMO_TAG_OVERHEAD_WIRE_SECTION, SUMO_TAG_TRACTION_SUBSTATION);
            } else {
                // build Overhead Wire
                GNEAdditional* overheadWire = new GNEOverheadWire(id, lanes, tractionSubstation, myNet, startPos, endPos, friendlyPos, forbiddenInnerLanes, parameters);
                // insert depending of allowUndoRedo
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::OVERHEADWIRE, "add " + toString(SUMO_TAG_OVERHEAD_WIRE_SECTION) + " '" + id + "'");
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(overheadWire, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    myNet->getAttributeCarriers()->insertAdditional(overheadWire);
                    for (const auto& lane : lanes) {
                        lane->addChildElement(overheadWire);
                    }
                    overheadWire->incRef("buildOverheadWire");
                }
            }
        } else {
            writeErrorInvalidLanes(SUMO_TAG_E2DETECTOR, id);
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_OVERHEAD_WIRE_SECTION, id);
    }
}


void
GNEAdditionalHandler::buildOverheadWireClamp(const CommonXMLStructure::SumoBaseObject* /* sumoBaseObject */, const std::string& /* id */, const std::string& /* overheadWireIDStartClamp */,
        const std::string& /* laneIDStartClamp */, const std::string& /* overheadWireIDEndClamp */, const std::string& /* laneIDEndClamp */,
        const Parameterised::Map& /* parameters */) {
    //
}


void
GNEAdditionalHandler::buildPolygon(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& type,
                                   const RGBColor& color, double layer, double angle, const std::string& imgFile, bool relativePath, const PositionVector& shape, bool geo, bool fill,
                                   double lineWidth, const std::string& name, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_POLY, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_POLY, id, false) != nullptr) {
        writeErrorDuplicated(SUMO_TAG_TAZ, id);
    } else if (myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TAZ, id, false) != nullptr) {
        writeErrorDuplicated(SUMO_TAG_TAZ, id);
    } else if (lineWidth < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POLY, id, SUMO_ATTR_LINEWIDTH);
    } else {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // create poly
        GNEPoly* poly = new GNEPoly(myNet, id, type, shape, geo, fill, lineWidth, color, layer, angle, imgFile, relativePath, name, parameters);
        // add it depending of allow undoRed
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::POLY, "add " + toString(SUMO_TAG_POLY) + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(poly, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert shape without allowing undo/redo
            myNet->getAttributeCarriers()->insertAdditional(poly);
            poly->incRef("addPolygon");
        }
    }
}


void
GNEAdditionalHandler::buildPOI(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& type,
                               const RGBColor& color, const double x, const double y, double layer, double angle, const std::string& imgFile, bool relativePath,
                               double width, double height, const std::string& name, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_POI, id);
    } else if (width < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_WIDTH);
    } else if (height < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_HEIGHT);
    } else if (!SUMOXMLDefinitions::isValidFilename(imgFile)) {
        writeErrorInvalidFilename(SUMO_TAG_POI, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_POI, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POILANE, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POIGEO, id, false) == nullptr)) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // create POI
        GNEPOI* POI = new GNEPOI(myNet, id, type, color, x, y, false, layer, angle, imgFile, relativePath, width, height, name, parameters);
        // add it depending of allow undoRed
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::POI, "add " + POI->getTagStr() + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(POI, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert shape without allowing undo/redo
            myNet->getAttributeCarriers()->insertAdditional(POI);
            POI->incRef("addPOI");
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_POI, id);
    }
}


void
GNEAdditionalHandler::buildPOILane(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& type,
                                   const RGBColor& color, const std::string& laneID, double posOverLane, const bool friendlyPos, double posLat, double layer, double angle,
                                   const std::string& imgFile, bool relativePath, double width, double height, const std::string& name, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_POI, id);
    } else if (width < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_WIDTH);
    } else if (height < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_HEIGHT);
    } else if (!SUMOXMLDefinitions::isValidFilename(imgFile)) {
        writeErrorInvalidFilename(SUMO_TAG_POI, id);
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_POI, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POILANE, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POIGEO, id, false) == nullptr)) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // get lane
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // check lane
        if (lane == nullptr) {
            writeErrorInvalidParent(SUMO_TAG_POI, SUMO_TAG_LANE);
        } else if (!checkLanePosition(posOverLane, 0, lane->getParentEdge()->getNBEdge()->getFinalLength(), friendlyPos)) {
            writeErrorInvalidPosition(SUMO_TAG_POI, id);
        } else {
            // create POI
            GNEAdditional* POILane = new GNEPOI(myNet, id, type, color, lane, posOverLane, friendlyPos, posLat, layer, angle, imgFile, relativePath, width, height, name, parameters);
            // add it depending of allow undoRed
            if (myAllowUndoRedo) {
                myNet->getViewNet()->getUndoList()->begin(GUIIcon::POILANE, "add " + POILane->getTagStr() + " '" + id + "'");
                myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(POILane, true), true);
                myNet->getViewNet()->getUndoList()->end();
            } else {
                // insert shape without allowing undo/redo
                myNet->getAttributeCarriers()->insertAdditional(POILane);
                lane->addChildElement(POILane);
                POILane->incRef("buildPOILane");
            }
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_POI, id);
    }
}


void
GNEAdditionalHandler::buildPOIGeo(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& id, const std::string& type,
                                  const RGBColor& color, const double lon, const double lat, double layer, double angle, const std::string& imgFile, bool relativePath,
                                  double width, double height, const std::string& name, const Parameterised::Map& parameters) {
    // check conditions
    if (!SUMOXMLDefinitions::isValidAdditionalID(id)) {
        writeInvalidID(SUMO_TAG_POI, id);
    } else if (width < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_WIDTH);
    } else if (height < 0) {
        writeErrorInvalidNegativeValue(SUMO_TAG_POI, id, SUMO_ATTR_HEIGHT);
    } else if (!SUMOXMLDefinitions::isValidFilename(imgFile)) {
        writeErrorInvalidFilename(SUMO_TAG_POI, id);
    } else if (GeoConvHelper::getFinal().getProjString() == "!") {
        WRITE_ERROR("Could not build " + toString(SUMO_TAG_POI) + " with ID '" + id + "' in netedit; Networ requires a geo projection.");
    } else if ((myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_POI, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POILANE, id, false) == nullptr) ||
               (myNet->getAttributeCarriers()->retrieveAdditional(GNE_TAG_POIGEO, id, false) == nullptr)) {
        // get NETEDIT parameters
        NeteditParameters neteditParameters(sumoBaseObject);
        // create POIGEO
        GNEPOI* POIGEO = new GNEPOI(myNet, id, type, color, lon, lat, true, layer, angle, imgFile, relativePath, width, height, name, parameters);
        // add it depending of allow undoRed
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::POIGEO, "add " + POIGEO->getTagStr() + " '" + id + "'");
            myNet->getViewNet()->getUndoList()->add(new GNEChange_Additional(POIGEO, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert shape without allowing undo/redo
            myNet->getAttributeCarriers()->insertAdditional(POIGEO);
            POIGEO->incRef("buildPOIGeo");
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_POI, id);
    }
}


bool
GNEAdditionalHandler::accessCanBeCreated(GNEAdditional* busStopParent, GNEEdge* edge) {
    // check if exist another acces for the same busStop in the given edge
    for (const auto& additional : busStopParent->getChildAdditionals()) {
        for (const auto& lane : edge->getLanes()) {
            if (additional->getAttribute(SUMO_ATTR_LANE) == lane->getID()) {
                return false;
            }
        }
    }
    return true;
}


bool
GNEAdditionalHandler::checkOverlappingRerouterIntervals(GNEAdditional* rerouter, SUMOTime newBegin, SUMOTime newEnd) {
    // declare a vector to keep sorted rerouter children
    std::vector<std::pair<SUMOTime, SUMOTime>> sortedIntervals;
    // iterate over child additional
    for (const auto& rerouterChild : rerouter->getChildAdditionals()) {
        if (!rerouterChild->getTagProperty().isSymbol()) {
            sortedIntervals.push_back(std::make_pair((SUMOTime)0., (SUMOTime)0.));
            // set begin and end
            sortedIntervals.back().first = TIME2STEPS(rerouterChild->getAttributeDouble(SUMO_ATTR_BEGIN));
            sortedIntervals.back().second = TIME2STEPS(rerouterChild->getAttributeDouble(SUMO_ATTR_END));
        }
    }
    // add new intervals
    sortedIntervals.push_back(std::make_pair(newBegin, newEnd));
    // sort children
    std::sort(sortedIntervals.begin(), sortedIntervals.end());
    // check overlapping after sorting
    for (int i = 0; i < (int)sortedIntervals.size() - 1; i++) {
        if (sortedIntervals.at(i).second > sortedIntervals.at(i + 1).first) {
            return false;
        }
    }
    return true;
}


bool
GNEAdditionalHandler::checkLanePosition(double pos, const double length, const double laneLength, const bool friendlyPos) {
    if (friendlyPos) {
        return true;
    }
    // adjust from and to (negative means that start at the end of lane and count backward)
    if (pos < 0) {
        pos += laneLength;
    }
    // check extremes
    if ((pos < 0) || (pos > laneLength)) {
        return false;
    }
    // check pos + length
    if ((pos + length) > laneLength) {
        return false;
    }
    // all OK
    return true;
}


void
GNEAdditionalHandler::fixLanePosition(double& pos, double& length, const double laneLength) {
    // negative pos means that start at the end of lane and count backward)
    if (pos < 0) {
        pos += laneLength;
    }
    // set position at the start
    if (pos < 0) {
        pos = 0;
    }
    // adjust pos
    if (pos >= laneLength) {
        pos = (laneLength - POSITION_EPS);
    }
    // adjust length
    if ((length < 0) || ((pos + length) > laneLength)) {
        length = POSITION_EPS;
    }
}


bool
GNEAdditionalHandler::checkLaneDoublePosition(double from, double to, const double laneLength, const bool friendlyPos) {
    if (friendlyPos) {
        return true;
    }
    // adjust from and to (negative means that start at the end of lane and count backward)
    if (from == INVALID_DOUBLE) {
        from = 0;
    }
    if (to == INVALID_DOUBLE) {
        to = laneLength;
    }
    if (from < 0) {
        from += laneLength;
    }
    if (to < 0) {
        to += laneLength;
    }
    if ((to - from) < POSITION_EPS) {
        return false;
    }
    if ((from < 0) || (from > laneLength)) {
        return false;
    }
    if ((to < 0) || (to > laneLength)) {
        return false;
    }
    return true;
}


void
GNEAdditionalHandler::fixLaneDoublePosition(double& from, double& to, const double laneLength) {
    // adjust from (negative means that start at the end of lane and count backward)
    if (from == INVALID_DOUBLE) {
        from = 0;
    }
    if (to == INVALID_DOUBLE) {
        to = laneLength;
    }
    if (from < 0) {
        from += laneLength;
    }
    if (from < 0) {
        from = 0;
    } else if (from > laneLength) {
        from = laneLength;
    }
    // adjust to
    if (to < 0) {
        to += laneLength;
    }
    if (to < 0) {
        to = 0;
    } else if (to > laneLength) {
        to = laneLength;
    }
    // to has more priorty as from, and distance between from and to must be >= POSITION_EPS
    if ((to - from) < POSITION_EPS) {
        if (to >= POSITION_EPS) {
            from = to - POSITION_EPS;
        } else {
            from = 0;
            to = POSITION_EPS;
        }
    }
}


bool
GNEAdditionalHandler::checkMultiLanePosition(double fromPos, const double fromLaneLength, const double toPos, const double tolaneLength, const bool friendlyPos) {
    if (friendlyPos) {
        return true;
    } else {
        return (checkLanePosition(fromPos, 0, fromLaneLength, false) && checkLanePosition(toPos, 0, tolaneLength, false));
    }
}


void
GNEAdditionalHandler::fixMultiLanePosition(double fromPos, const double fromLaneLength, double toPos, const double tolaneLength) {
    double length = 0;
    fixLanePosition(fromPos, length, fromLaneLength);
    fixLanePosition(toPos, length, tolaneLength);
}


void
GNEAdditionalHandler::writeInvalidID(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; ID contains invalid characters.");
}


void
GNEAdditionalHandler::writeErrorInvalidPosition(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; Invalid position over lane.");
}


void
GNEAdditionalHandler::writeErrorDuplicated(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; declared twice.");
}


void
GNEAdditionalHandler::writeErrorInvalidParent(const SumoXMLTag tag, const SumoXMLTag parent) const {
    WRITE_ERROR("Could not build " + toString(tag) + " in netedit; " +  toString(parent) + " doesn't exist.");
}


void
GNEAdditionalHandler::writeErrorInvalidNegativeValue(const SumoXMLTag tag, const std::string& id, const SumoXMLAttr attribute) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; attribute " +  toString(attribute) + " cannot be negative.");
}


void
GNEAdditionalHandler::writeErrorInvalidVTypes(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; list of VTypes isn't valid.");
}


void
GNEAdditionalHandler::writeErrorInvalidFilename(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; filename is invalid.");
}


void
GNEAdditionalHandler::writeErrorInvalidLanes(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; list of lanes isn't valid.");
}


bool
GNEAdditionalHandler::checkListOfVehicleTypes(const std::vector<std::string>& vTypeIDs) const {
    for (const auto& vTypeID : vTypeIDs) {
        if (!SUMOXMLDefinitions::isValidTypeID(vTypeID)) {
            return false;
        }
    }
    return true;
}


GNEAdditional*
GNEAdditionalHandler::getAdditionalParent(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, SumoXMLTag tag) const {
    if (sumoBaseObject->getParentSumoBaseObject() == nullptr) {
        return nullptr;
    } else if (!sumoBaseObject->getParentSumoBaseObject()->hasStringAttribute(SUMO_ATTR_ID)) {
        return nullptr;
    } else {
        return myNet->getAttributeCarriers()->retrieveAdditional(tag, sumoBaseObject->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID), false);
    }
}


GNEAdditional*
GNEAdditionalHandler::getRerouterIntervalParent(const CommonXMLStructure::SumoBaseObject* sumoBaseObject) const {
    if (sumoBaseObject->getParentSumoBaseObject() == nullptr) {
        // parent interval doesn't exist
        return nullptr;
    } else if (sumoBaseObject->getParentSumoBaseObject()->getParentSumoBaseObject() == nullptr) {
        // rerouter parent doesn't exist
        return nullptr;
    } else if (!sumoBaseObject->getParentSumoBaseObject()->getParentSumoBaseObject()->hasStringAttribute(SUMO_ATTR_ID) ||   // rerouter ID
               !sumoBaseObject->getParentSumoBaseObject()->hasTimeAttribute(SUMO_ATTR_BEGIN) || // interval begin
               !sumoBaseObject->getParentSumoBaseObject()->hasTimeAttribute(SUMO_ATTR_END)) {   // interval end
        return nullptr;
    } else {
        return myNet->getAttributeCarriers()->retrieveRerouterInterval(
                   sumoBaseObject->getParentSumoBaseObject()->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID),
                   sumoBaseObject->getParentSumoBaseObject()->getTimeAttribute(SUMO_ATTR_BEGIN),
                   sumoBaseObject->getParentSumoBaseObject()->getTimeAttribute(SUMO_ATTR_END));
    }
}


std::vector<GNEEdge*>
GNEAdditionalHandler::parseEdges(const SumoXMLTag tag, const std::vector<std::string>& edgeIDs) const {
    std::vector<GNEEdge*> edges;
    for (const auto& edgeID : edgeIDs) {
        GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
        // empty edges aren't allowed. If edge is empty, write error, clear edges and stop
        if (edge == nullptr) {
            WRITE_ERROR("Could not build " + toString(tag) + " in netedit; " +  toString(SUMO_TAG_EDGE) + " doesn't exist.");
            edges.clear();
            return edges;
        } else {
            edges.push_back(edge);
        }
    }
    return edges;
}


std::vector<GNELane*>
GNEAdditionalHandler::parseLanes(const SumoXMLTag tag, const std::vector<std::string>& laneIDs) const {
    std::vector<GNELane*> lanes;
    for (const auto& laneID : laneIDs) {
        GNELane* lane = myNet->getAttributeCarriers()->retrieveLane(laneID, false);
        // empty lanes aren't allowed. If lane is empty, write error, clear lanes and stop
        if (lane == nullptr) {
            WRITE_ERROR("Could not build " + toString(tag) + " in netedit; " +  toString(SUMO_TAG_LANE) + " doesn't exist.");
            lanes.clear();
            return lanes;
        } else {
            lanes.push_back(lane);
        }
    }
    return lanes;
}


GNEAdditionalHandler::GNEAdditionalHandler() :
    myNet(nullptr),
    myAllowUndoRedo(false) {
}

// ===========================================================================
// GNEAdditionalHandler::NeteditParameters method definitions
// ===========================================================================

GNEAdditionalHandler::NeteditParameters::NeteditParameters(const CommonXMLStructure::SumoBaseObject* sumoBaseObject) :
    select(sumoBaseObject->hasBoolAttribute(GNE_ATTR_SELECTED) ? sumoBaseObject->getBoolAttribute(GNE_ATTR_SELECTED) : false),
    centerAfterCreation(sumoBaseObject->hasBoolAttribute(GNE_ATTR_CENTER_AFTER_CREATION) ? sumoBaseObject->getBoolAttribute(GNE_ATTR_CENTER_AFTER_CREATION) : false) {
}


GNEAdditionalHandler::NeteditParameters::~NeteditParameters() {}


GNEAdditionalHandler::NeteditParameters::NeteditParameters() :
    select(false),
    centerAfterCreation(false) {
}

/****************************************************************************/
