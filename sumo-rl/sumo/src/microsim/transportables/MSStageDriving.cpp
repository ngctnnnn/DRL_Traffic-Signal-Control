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
/// @file    MSStageDriving.cpp
/// @author  Melanie Weber
/// @author  Andreas Kendziorra
/// @author  Michael Behrisch
/// @date    Thu, 12 Jun 2014
///
// The common superclass for modelling transportable objects like persons and containers
/****************************************************************************/
#include <config.h>

#include <utils/common/StringTokenizer.h>
#include <utils/geom/GeomHelper.h>
#include <utils/vehicle/SUMOVehicleParameter.h>
#include <utils/router/PedestrianRouter.h>
#include <utils/router/IntermodalRouter.h>
#include <microsim/MSEdge.h>
#include <microsim/MSLane.h>
#include <microsim/MSNet.h>
#include <microsim/MSInsertionControl.h>
#include <microsim/MSVehicleControl.h>
#include <microsim/MSStoppingPlace.h>
#include <microsim/transportables/MSPerson.h>
#include <microsim/devices/MSTransportableDevice.h>
#include <microsim/devices/MSDevice_Taxi.h>
#include <microsim/devices/MSDevice_Tripinfo.h>
#include <microsim/transportables/MSTransportableControl.h>
#include <microsim/transportables/MSStageDriving.h>
#include <microsim/transportables/MSPModel.h>


// ===========================================================================
// method definitions
// ===========================================================================
MSStageDriving::MSStageDriving(const MSEdge* origin, const MSEdge* destination,
                               MSStoppingPlace* toStop, const double arrivalPos, const std::vector<std::string>& lines,
                               const std::string& group,
                               const std::string& intendedVeh, SUMOTime intendedDepart) :
    MSStage(destination, toStop, arrivalPos, MSStageType::DRIVING, group),
    myOrigin(origin),
    myLines(lines.begin(), lines.end()),
    myVehicle(nullptr),
    myVehicleID("NULL"),
    myVehicleVClass(SVC_IGNORING),
    myVehicleDistance(-1.),
    myTimeLoss(-1),
    myWaitingSince(-1),
    myWaitingEdge(nullptr),
    myStopWaitPos(Position::INVALID),
    myOriginStop(nullptr),
    myIntendedVehicleID(intendedVeh),
    myIntendedDepart(intendedDepart) {
}


MSStage*
MSStageDriving::clone() const {
    return new MSStageDriving(myOrigin, myDestination, myDestinationStop, myArrivalPos, std::vector<std::string>(myLines.begin(), myLines.end()),
                              myGroup, myIntendedVehicleID, myIntendedDepart);
}


MSStageDriving::~MSStageDriving() {}


const MSEdge*
MSStageDriving::getEdge() const {
    if (myVehicle != nullptr) {
        if (myVehicle->getLane() != nullptr) {
            return &myVehicle->getLane()->getEdge();
        }
        return myVehicle->getEdge();
    } else if (myArrived >= 0) {
        return myDestination;
    } else {
        return myWaitingEdge;
    }
}


const MSEdge*
MSStageDriving::getFromEdge() const {
    return myWaitingEdge;
}


double
MSStageDriving::getEdgePos(SUMOTime /* now */) const {
    if (isWaiting4Vehicle()) {
        return myWaitingPos;
    } else if (myArrived >= 0) {
        return myArrivalPos;
    } else {
        // vehicle may already have passed the lane (check whether this is correct)
        return MIN2(myVehicle->getPositionOnLane(), getEdge()->getLength());
    }
}

int
MSStageDriving::getDirection() const {
    if (isWaiting4Vehicle()) {
        return MSPModel::UNDEFINED_DIRECTION;
    } else if (myArrived >= 0) {
        return MSPModel::UNDEFINED_DIRECTION;
    } else {
        return MSPModel::FORWARD;
    }
}

const MSLane*
MSStageDriving::getLane() const {
    return myVehicle != nullptr ? myVehicle->getLane() : nullptr;
}


Position
MSStageDriving::getPosition(SUMOTime /* now */) const {
    if (isWaiting4Vehicle()) {
        if (myStopWaitPos != Position::INVALID) {
            return myStopWaitPos;
        }
        return getEdgePosition(myWaitingEdge, myWaitingPos,
                               ROADSIDE_OFFSET * (MSGlobals::gLefthand ? -1 : 1));
    } else if (myArrived >= 0) {
        return getEdgePosition(myDestination, myArrivalPos,
                               ROADSIDE_OFFSET * (MSGlobals::gLefthand ? -1 : 1));
    } else {
        return myVehicle->getPosition();
    }
}


double
MSStageDriving::getAngle(SUMOTime /* now */) const {
    if (isWaiting4Vehicle()) {
        return getEdgeAngle(myWaitingEdge, myWaitingPos) + M_PI / 2. * (MSGlobals::gLefthand ? -1 : 1);
    } else if (myArrived >= 0) {
        return getEdgeAngle(myDestination, myArrivalPos) + M_PI / 2. * (MSGlobals::gLefthand ? -1 : 1);
    } else {
        MSVehicle* veh = dynamic_cast<MSVehicle*>(myVehicle);
        if (veh != nullptr) {
            return veh->getAngle();
        } else {
            return 0;
        }
    }
}


double
MSStageDriving::getDistance() const {
    if (myVehicle != nullptr) {
        // distance was previously set to driven distance upon embarking
        return myVehicle->getOdometer() - myVehicleDistance;
    }
    return myVehicleDistance;
}


std::string
MSStageDriving::getStageDescription(const bool isPerson) const {
    return isWaiting4Vehicle() ? "waiting for " + joinToString(myLines, ",") : (isPerson ? "driving" : "transport");
}


std::string
MSStageDriving::getStageSummary(const bool isPerson) const {
    const std::string dest = (getDestinationStop() == nullptr ?
                              " edge '" + getDestination()->getID() + "'" :
                              " stop '" + getDestinationStop()->getID() + "'" + (
                                  getDestinationStop()->getMyName() != "" ? " (" + getDestinationStop()->getMyName() + ")" : ""));
    const std::string intended = myIntendedVehicleID != "" ?
                                 " (vehicle " + myIntendedVehicleID + " at time=" + time2string(myIntendedDepart) + ")" :
                                 "";
    const std::string modeName = isPerson ? "driving" : "transported";
    return isWaiting4Vehicle() ?
           "waiting for " + joinToString(myLines, ",") + intended + " then " + modeName + " to " + dest :
           modeName + " to " + dest;
}


void
MSStageDriving::proceed(MSNet* net, MSTransportable* transportable, SUMOTime now, MSStage* previous) {
    myOriginStop = (previous->getStageType() == MSStageType::TRIP
                    ? previous->getOriginStop()
                    : previous->getDestinationStop());
    myWaitingSince = now;
    const bool isPerson = transportable->isPerson();
    if (transportable->getParameter().departProcedure == DepartDefinition::TRIGGERED
            && transportable->getNumRemainingStages() == transportable->getNumStages() - 1) {
        // we are the first real stage (stage 0 is WAITING_FOR_DEPART)
        const std::string vehID = *myLines.begin();
        SUMOVehicle* startVeh = net->getVehicleControl().getVehicle(vehID);
        if (startVeh == nullptr) {
            throw ProcessError("Vehicle '" + vehID + "' not found for triggered departure of " +
                               (isPerson ? "person" : "container") + " '" + transportable->getID() + "'.");
        }
        setVehicle(startVeh);
        if (myOriginStop != nullptr) {
            myOriginStop->removeTransportable(transportable);
        }
        myWaitingEdge = previous->getEdge();
        myStopWaitPos = Position::INVALID;
        myWaitingPos = previous->getEdgePos(now);
        myVehicle->addTransportable(transportable);
        return;
    }
    if (myOriginStop != nullptr) {
        // the arrival stop may have an access point
        myWaitingEdge = &myOriginStop->getLane().getEdge();
        myStopWaitPos = myOriginStop->getWaitPosition(transportable);
        myWaitingPos = myOriginStop->getWaitingPositionOnLane(transportable);
    } else {
        myWaitingEdge = previous->getEdge();
        myStopWaitPos = Position::INVALID;
        myWaitingPos = previous->getEdgePos(now);
    }
    if (myOrigin != nullptr && myOrigin != myWaitingEdge) {
        // transfer at junction
        myWaitingEdge = myOrigin;
        myWaitingPos = 0;
    }
    SUMOVehicle* const availableVehicle = myWaitingEdge->getWaitingVehicle(transportable, myWaitingPos);
    const bool triggered = availableVehicle != nullptr &&
                           ((isPerson && availableVehicle->getParameter().departProcedure == DepartDefinition::TRIGGERED) ||
                            (!isPerson && availableVehicle->getParameter().departProcedure == DepartDefinition::CONTAINER_TRIGGERED));
    if (triggered && !availableVehicle->hasDeparted()) {
        setVehicle(availableVehicle);
        if (myOriginStop != nullptr) {
            myOriginStop->removeTransportable(transportable);
        }
        myVehicle->addTransportable(transportable);
        net->getInsertionControl().add(myVehicle);
        if (myVehicle->getEdge()->isTazConnector()) {
            for (MSEdge* out : myVehicle->getEdge()->getSuccessors()) {
                out->removeWaiting(myVehicle);
            }
        } else {
            myWaitingEdge->removeWaiting(myVehicle);
        }
        net->getVehicleControl().unregisterOneWaiting();
    } else {
        registerWaiting(transportable, now);
    }
}

void
MSStageDriving::registerWaiting(MSTransportable* transportable, SUMOTime now) {
    // check if the ride can be conducted and reserve it
    if (MSDevice_Taxi::isReservation(getLines())) {
        const MSEdge* to = getDestination();
        double toPos = getArrivalPos();
        if ((to->getPermissions() & SVC_TAXI) == 0 && getDestinationStop() != nullptr) {
            // try to find usable access edge
            for (const auto& tuple : getDestinationStop()->getAllAccessPos()) {
                const MSEdge* access = &std::get<0>(tuple)->getEdge();
                if ((access->getPermissions() & SVC_TAXI) != 0) {
                    to = access;
                    toPos = std::get<1>(tuple);
                    break;
                }
            }
        }
        if ((myWaitingEdge->getPermissions() & SVC_TAXI) == 0 && myOriginStop != nullptr) {
            // try to find usable access edge
            for (const auto& tuple : myOriginStop->getAllAccessPos()) {
                const MSEdge* access = &std::get<0>(tuple)->getEdge();
                if ((access->getPermissions() & SVC_TAXI) != 0) {
                    myWaitingEdge = access;
                    myStopWaitPos = Position::INVALID;
                    myWaitingPos = std::get<1>(tuple);
                    break;
                }
            }
        }
        MSDevice_Taxi::addReservation(transportable, getLines(), now, now, myWaitingEdge, myWaitingPos, to, toPos, myGroup);
    }
    if (transportable->isPerson()) {
        MSNet::getInstance()->getPersonControl().addWaiting(myWaitingEdge, transportable);
    } else {
        MSNet::getInstance()->getContainerControl().addWaiting(myWaitingEdge, transportable);
    }
    myWaitingEdge->addTransportable(transportable);
}

void
MSStageDriving::tripInfoOutput(OutputDevice& os, const MSTransportable* const transportable) const {
    const SUMOTime now = MSNet::getInstance()->getCurrentTimeStep();
    const SUMOTime departed = myDeparted >= 0 ? myDeparted : now;
    const SUMOTime waitingTime = myWaitingSince >= 0 ? departed - myWaitingSince : -1;
    const SUMOTime duration = myArrived - myDeparted;
    MSDevice_Tripinfo::addRideTransportData(transportable->isPerson(), myVehicleDistance, duration, myVehicleVClass, myVehicleLine, waitingTime);
    os.openTag(transportable->isPerson() ? "ride" : "transport");
    os.writeAttr("waitingTime", waitingTime >= 0 ? time2string(waitingTime) : "-1");
    os.writeAttr("vehicle", myVehicleID);
    os.writeAttr("depart", myDeparted >= 0 ? time2string(myDeparted) : "-1");
    os.writeAttr("arrival", myArrived >= 0 ? time2string(myArrived) : "-1");
    os.writeAttr("arrivalPos", myArrived >= 0 ? toString(getArrivalPos()) : "-1");
    os.writeAttr("duration", myArrived >= 0 ? time2string(duration) :
                 (myDeparted >= 0 ? time2string(now - myDeparted) : "-1"));
    os.writeAttr("routeLength", myArrived >= 0 || myVehicle != nullptr ? toString(getDistance()) : "-1");
    os.writeAttr("timeLoss", myArrived >= 0 ? time2string(myTimeLoss) : "-1");
    os.closeTag();
}


void
MSStageDriving::routeOutput(const bool isPerson, OutputDevice& os, const bool withRouteLength, const MSStage* const previous) const {
    os.openTag(isPerson ? SUMO_TAG_RIDE : SUMO_TAG_TRANSPORT);
    if (getFromEdge() != nullptr) {
        os.writeAttr(SUMO_ATTR_FROM, getFromEdge()->getID());
    } else if (previous != nullptr && previous->getStageType() == MSStageType::WAITING_FOR_DEPART) {
        os.writeAttr(SUMO_ATTR_FROM, previous->getEdge()->getID());
    }
    os.writeAttr(SUMO_ATTR_TO, getDestination()->getID());
    std::string comment = "";
    if (myDestinationStop != nullptr) {
        os.writeAttr(toString(myDestinationStop->getElement()), myDestinationStop->getID());
        if (myDestinationStop->getMyName() != "") {
            comment = " <!-- " + StringUtils::escapeXML(myDestinationStop->getMyName(), true) + " -->";
        }
    } else if (!unspecifiedArrivalPos()) {
        os.writeAttr(SUMO_ATTR_ARRIVALPOS, myArrivalPos);
    }
    os.writeAttr(SUMO_ATTR_LINES, myLines);
    if (myIntendedVehicleID != "") {
        os.writeAttr(SUMO_ATTR_INTENDED, myIntendedVehicleID);
    }
    if (myIntendedDepart >= 0) {
        os.writeAttr(SUMO_ATTR_DEPART, time2string(myIntendedDepart));
    }
    if (withRouteLength) {
        os.writeAttr("routeLength", myVehicleDistance);
    }
    if (OptionsCont::getOptions().getBool("vehroute-output.exit-times")) {
        os.writeAttr(SUMO_ATTR_STARTED, myDeparted >= 0 ? time2string(myDeparted) : "-1");
        os.writeAttr(SUMO_ATTR_ENDED, myArrived >= 0 ? time2string(myArrived) : "-1");
    }
    os.closeTag(comment);
}


bool
MSStageDriving::isWaitingFor(const SUMOVehicle* vehicle) const {
    assert(myLines.size() > 0);
    return (myLines.count(vehicle->getID()) > 0
            || myLines.count(vehicle->getParameter().line) > 0
            || MSDevice_Taxi::compatibleLine(vehicle->getParameter().line, *myLines.begin())
            || (myLines.count("ANY") > 0 && (
                    myDestinationStop == nullptr
                    ? vehicle->stopsAtEdge(myDestination)
                    : vehicle->stopsAt(myDestinationStop))));
}


bool
MSStageDriving::isWaiting4Vehicle() const {
    return myVehicle == nullptr && myArrived < 0;
}


SUMOTime
MSStageDriving::getWaitingTime(SUMOTime now) const {
    return isWaiting4Vehicle() ? now - myWaitingSince : 0;
}


double
MSStageDriving::getSpeed() const {
    return myVehicle == nullptr ? 0 : myVehicle->getSpeed();
}


ConstMSEdgeVector
MSStageDriving::getEdges() const {
    ConstMSEdgeVector result;
    result.push_back(getFromEdge());
    result.push_back(getDestination());
    return result;
}

double
MSStageDriving::getArrivalPos() const {
    return unspecifiedArrivalPos() ? getDestination()->getLength() : myArrivalPos;
}

bool
MSStageDriving::unspecifiedArrivalPos() const {
    return myArrivalPos == std::numeric_limits<double>::infinity();
}

const std::string
MSStageDriving::setArrived(MSNet* net, MSTransportable* transportable, SUMOTime now, const bool vehicleArrived) {
    MSStage::setArrived(net, transportable, now, vehicleArrived);
    if (myVehicle != nullptr) {
        // distance was previously set to driven distance upon embarking
        myVehicleDistance = myVehicle->getOdometer() - myVehicleDistance;
        myTimeLoss = myVehicle->getTimeLoss() - myTimeLoss;
        if (vehicleArrived) {
            myArrivalPos = myVehicle->getArrivalPos();
        } else {
            myArrivalPos = myVehicle->getPositionOnLane();
        }
    } else {
        myVehicleDistance = -1.;
        myTimeLoss = -1;
    }
    myVehicle = nullptr; // avoid dangling pointer after vehicle arrival
    return "";
}


void
MSStageDriving::setVehicle(SUMOVehicle* v) {
    myVehicle = v;
    if (myVehicle != nullptr) {
        myVehicleID = v->getID();
        myVehicleLine = v->getParameter().line;
        myVehicleType = v->getVehicleType().getID();
        myVehicleVClass = v->getVClass();
        if (myVehicle->hasDeparted()) {
            myVehicleDistance = myVehicle->getOdometer();
            myTimeLoss = myVehicle->getTimeLoss();
        } else {
            // it probably got triggered by the person
            myVehicleDistance = 0.;
            myTimeLoss = 0;
        }
    }
}

void
MSStageDriving::abort(MSTransportable* t) {
    myDestinationStop = nullptr;
    if (myVehicle != nullptr) {
        // jumping out of a moving vehicle!
        myVehicle->removeTransportable(t);
        myDestination = myVehicle->getLane() == nullptr ? myVehicle->getEdge() : &myVehicle->getLane()->getEdge();
        // myVehicleDistance and myTimeLoss are updated in subsequent call to setArrived
    } else {
        MSTransportableControl& tc = (t->isPerson() ?
                                      MSNet::getInstance()->getPersonControl() :
                                      MSNet::getInstance()->getContainerControl());
        tc.abortWaitingForVehicle(t);
        MSDevice_Taxi::removeReservation(t, getLines(), myWaitingEdge, myWaitingPos, myDestination, getArrivalPos(), myGroup);
        myDestination = myWaitingEdge;
    }
}


std::string
MSStageDriving::getWaitingDescription() const {
    return isWaiting4Vehicle() ? ("waiting for " + joinToString(myLines, ",")
                                  + " at " + (myDestinationStop == nullptr
                                          ? ("edge '" + myWaitingEdge->getID() + "'")
                                          : ("busStop '" + myDestinationStop->getID() + "'"))
                                 ) : "";
}


bool
MSStageDriving::canLeaveVehicle(const MSTransportable* t, const SUMOVehicle& veh) {
    bool canLeave = false;
    if (t->getDestination() == veh.getEdge()) {
        // if this is the last stage, we can use the arrivalPos of the person
        const bool unspecifiedAP = unspecifiedArrivalPos() && (
                                       t->getNumRemainingStages() > 1 || !t->getParameter().wasSet(VEHPARS_ARRIVALPOS_SET));
        const double arrivalPos = (unspecifiedArrivalPos()
                                   ? SUMOVehicleParameter::interpretEdgePos(t->getParameter().arrivalPos, veh.getEdge()->getLength(),
                                           SUMO_ATTR_ARRIVALPOS, t->getID(), true)
                                   : getArrivalPos());
        if (unspecifiedAP ||
                veh.isStoppedInRange(arrivalPos, veh.getLength() + MSGlobals::gStopTolerance)) {
            canLeave = true;
        }
    }
    if (myDestinationStop != nullptr) {
        if (!canLeave) {
            // check with more tolerance due to busStop size and also check
            // access edges
            const double accessPos = myDestinationStop->getAccessPos(veh.getEdge());
            if (accessPos >= 0) {
                double tolerance = veh.getLength() + MSGlobals::gStopTolerance;
                if (&myDestinationStop->getLane().getEdge() == veh.getEdge()) {
                    // accessPos is in the middle of the stop
                    tolerance += (myDestinationStop->getEndLanePosition() - myDestinationStop->getBeginLanePosition()) / 2;
                }
                canLeave = veh.isStoppedInRange(accessPos, tolerance);
            }
        }
    }
    return canLeave;
}


void
MSStageDriving::saveState(std::ostringstream& out) {
    const bool hasVehicle = myVehicle != nullptr;
    out << " " << myWaitingSince << " " << myTimeLoss << " " << myArrived << " " << hasVehicle;
    if (hasVehicle) {
        out << " " << myDeparted << " " << myVehicle->getID() << " " << myVehicleDistance;
    }
}


void
MSStageDriving::loadState(MSTransportable* transportable, std::istringstream& state) {
    bool hasVehicle;
    state >> myWaitingSince >> myTimeLoss >> myArrived >> hasVehicle;
    if (hasVehicle) {
        std::string vehID;
        state >> myDeparted >> vehID;
        SUMOVehicle* startVeh = MSNet::getInstance()->getVehicleControl().getVehicle(vehID);
        setVehicle(startVeh);
        myVehicle->addTransportable(transportable);
        state >> myVehicleDistance;
    } else {
        // there should always be at least one prior WAITING_FOR_DEPART stage
        MSStage* previous = transportable->getNextStage(-1);
        myOriginStop = (previous->getStageType() == MSStageType::TRIP
                        ? previous->getOriginStop()
                        : previous->getDestinationStop());
        if (myOriginStop != nullptr) {
            // the arrival stop may have an access point
            myOriginStop->addTransportable(transportable);
            myWaitingEdge = &myOriginStop->getLane().getEdge();
            myStopWaitPos = myOriginStop->getWaitPosition(transportable);
            myWaitingPos = myOriginStop->getWaitingPositionOnLane(transportable);
        } else {
            myWaitingEdge = previous->getEdge();
            myStopWaitPos = Position::INVALID;
            myWaitingPos = previous->getArrivalPos();
        }
        registerWaiting(transportable, SIMSTEP);
    }
}


/****************************************************************************/
