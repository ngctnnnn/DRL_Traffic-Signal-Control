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
/// @file    MSLink.cpp
/// @author  Daniel Krajzewicz
/// @author  Jakob Erdmann
/// @author  Michael Behrisch
/// @author  Laura Bieker
/// @date    Sept 2002
///
// A connnection between lanes
/****************************************************************************/
#include <config.h>

#include <iostream>
#include <algorithm>
#include <limits>
#include <utils/iodevices/OutputDevice.h>
#include <utils/common/RandHelper.h>
#include <utils/common/StringTokenizer.h>
#include "MSNet.h"
#include "MSJunction.h"
#include "MSLink.h"
#include "MSLane.h"
#include <microsim/transportables/MSPerson.h>
#include <microsim/transportables/MSTransportableControl.h>
#include "MSEdge.h"
#include "MSGlobals.h"
#include "MSVehicle.h"
#include <microsim/lcmodels/MSAbstractLaneChangeModel.h>
#include <microsim/transportables/MSPModel.h>

//#define MSLink_DEBUG_CROSSING_POINTS
//#define MSLink_DEBUG_CROSSING_POINTS_DETAILS
//#define MSLink_DEBUG_OPENED
//#define DEBUG_APPROACHING
//#define DEBUG_ZIPPER
//#define DEBUG_WALKINGAREA
//#define DEBUG_COND (myLane->getID()=="43[0]_0" && myLaneBefore->getID()==":33_0_0")
//#define DEBUG_COND (myLane->getID()=="end_0")
//#define DEBUG_COND (true)
#define DEBUG_COND2(obj) (obj->isSelected())
//#define DEBUG_COND2(obj) (obj->getID() == "train2")
//#define DEBUG_COND_ZIPPER (gDebugFlag1)
//#define DEBUG_COND_ZIPPER (true)
#define DEBUG_COND_ZIPPER (ego->isSelected())

// ===========================================================================
// static member variables
// ===========================================================================
const SUMOTime MSLink::myLookaheadTime = TIME2STEPS(1);
// additional caution is needed when approaching a zipper link
const SUMOTime MSLink::myLookaheadTimeZipper = TIME2STEPS(4);

#define INVALID_TIME -1000

// the default safety gap when passing before oncoming pedestrians
#define JM_CROSSING_GAP_DEFAULT 10

// minimim width between sibling lanes to qualify as non-overlapping
#define DIVERGENCE_MIN_WIDTH 2.5

#define NO_INTERSECTION 10000.0

// ===========================================================================
// member method definitions
// ===========================================================================
MSLink::MSLink(MSLane* predLane, MSLane* succLane, MSLane* via, LinkDirection dir, LinkState state,
               double length, double foeVisibilityDistance, bool keepClear,
               MSTrafficLightLogic* logic, int tlIndex,
               bool indirect) :
    myLane(succLane),
    myLaneBefore(predLane),
    myIndex(-1),
    myTLIndex(tlIndex),
    myLogic(logic),
    myState(state),
    myLastGreenState(LINKSTATE_TL_GREEN_MINOR),
    myOffState(state),
    myLastStateChange(SUMOTime_MIN / 2), // a large negative value, but avoid overflows when subtracting
    myDirection(dir),
    myLength(length),
    myFoeVisibilityDistance(foeVisibilityDistance),
    myHasFoes(false),
    myAmCont(false),
    myAmContOff(false),
    myKeepClear(keepClear),
    myInternalLane(via),
    myInternalLaneBefore(nullptr),
    myMesoTLSPenalty(0),
    myGreenFraction(1),
    myLateralShift(0),
    myWalkingAreaFoe(nullptr),
    myWalkingAreaFoeExit(nullptr),
    myHavePedestrianCrossingFoe(false),
    myParallelRight(nullptr),
    myParallelLeft(nullptr),
    myAmIndirect(indirect),
    myRadius(std::numeric_limits<double>::max()),
    myJunction(nullptr) {

    if (MSGlobals::gLateralResolution > 0) {
        // detect lateral shift from lane geometries
        //std::cout << "DEBUG link=" << myLaneBefore->getID() << "->" << getViaLaneOrLane()->getID() << " hasInternal=" << MSNet::getInstance()->hasInternalLinks() << " shapeBefore=" << myLaneBefore->getShape().back() << " shapeFront=" << getViaLaneOrLane()->getShape().front() << "\n";
        if ((myInternalLane != nullptr || predLane->isInternal())
                && myLaneBefore->getShape().back() != getViaLaneOrLane()->getShape().front()) {
            PositionVector from = myLaneBefore->getShape();
            const PositionVector& to = getViaLaneOrLane()->getShape();
            const double dist = from.back().distanceTo2D(to.front());
            // figure out direction of shift
            try {
                from.move2side(dist);
            } catch (InvalidArgument&) {
            }
            myLateralShift = (from.back().distanceTo2D(to.front()) < dist) ? dist : -dist;
            if (MSGlobals::gLefthand) {
                myLateralShift *= -1;
            }
            //std::cout << " lateral shift link=" << myLaneBefore->getID() << "->" << getViaLaneOrLane()->getID() << " dist=" << dist << " shift=" << myLateralShift << "\n";
        }
    }
}


MSLink::~MSLink() {}


void
MSLink::setRequestInformation(int index, bool hasFoes, bool isCont,
                              const std::vector<MSLink*>& foeLinks,
                              const std::vector<MSLane*>& foeLanes,
                              MSLane* internalLaneBefore) {
//#ifdef MSLink_DEBUG_CROSSING_POINTS
//    std::cout << " setRequestInformation() for junction " << getViaLaneOrLane()->getEdge().getFromJunction()->getID()
//            << "\nInternalLanes = " << toString(getViaLaneOrLane()->getEdge().getFromJunction()->getInternalLanes())
//            << std::endl;
//#endif
    myIndex = index;
    myHasFoes = hasFoes;
    myAmCont = isCont;
    myFoeLinks = foeLinks;
    for (std::vector<MSLane*>::const_iterator it_lane = foeLanes.begin(); it_lane != foeLanes.end(); ++it_lane) {
        // cannot assign vector due to const-ness
        myFoeLanes.push_back(*it_lane);
    }
    myJunction = const_cast<MSJunction*>(myLane->getEdge().getFromJunction()); // junctionGraph is initialized after the whole network is loaded
    myAmContOff = isCont && myLogic != nullptr && internalLaneBefore == nullptr && checkContOff();
    myInternalLaneBefore = internalLaneBefore;
    MSLane* lane = nullptr;
    if (internalLaneBefore != nullptr) {
        // this is an exit link. compute crossing points with all foeLanes
        lane = internalLaneBefore;
        //} else if (myLane->getEdge().isCrossing()) {
        //    // this is the link to a pedestrian crossing. compute crossing points with all foeLanes
        //    // @note not currently used by pedestrians
        //    lane = myLane;
    }
#ifdef MSLink_DEBUG_CROSSING_POINTS
    std::cout << "link " << myIndex << " to " << getViaLaneOrLane()->getID() << " internalLaneBefore=" << (lane == 0 ? "NULL" : lane->getID()) << " has foes: " << toString(foeLanes) << "\n";
#endif
    if (lane != nullptr) {
        const bool beforeInternalJunction = lane->getLinkCont()[0]->getViaLaneOrLane()->getEdge().isInternal();
        if (lane->getIncomingLanes().size() != 1) {
            throw ProcessError("Internal lane '" + lane->getID() + "' has " + toString(lane->getIncomingLanes().size()) + " predecessors");
        }
        // compute crossing points
        for (std::vector<const MSLane*>::const_iterator it_lane = myFoeLanes.begin(); it_lane != myFoeLanes.end(); ++it_lane) {
            myHavePedestrianCrossingFoe = myHavePedestrianCrossingFoe || (*it_lane)->getEdge().isCrossing();
            const bool sameTarget = myLane == (*it_lane)->getLinkCont()[0]->getLane();
            if (sameTarget && !beforeInternalJunction && !contIntersect(lane, *it_lane)) {
                //if (myLane == (*it_lane)->getLinkCont()[0]->getLane()) {
                // this foeLane has the same target and merges at the end (lane exits the junction)
                const MSLane* sibling = *it_lane;
                const double minDist = MIN2(DIVERGENCE_MIN_WIDTH, 0.5 * (lane->getWidth() + sibling->getWidth()));
                if (lane->getShape().back().distanceTo2D(sibling->getShape().back()) >= minDist) {
                    // account for lateral shift by the entry links
                    if (sibling->getEntryLink()->isIndirect()) {
                        myLengthsBehindCrossing.push_back(std::make_pair(-NO_INTERSECTION, -NO_INTERSECTION)); // dummy value, never used
#ifdef MSLink_DEBUG_CROSSING_POINTS
                        std::cout << " " << lane->getID() << " dummy merge with indirect" << (*it_lane)->getID() << "\n";
#endif
                    } else {
                        myLengthsBehindCrossing.push_back(std::make_pair(0, 0)); // dummy value, never used
#ifdef MSLink_DEBUG_CROSSING_POINTS
                        std::cout << " " << lane->getID() << " dummy merge with " << (*it_lane)->getID() << "\n";
#endif
                    }
                } else {
                    const double distAfterDivergence = computeDistToDivergence(lane, sibling, minDist, false);
                    const double lbcLane = lane->interpolateGeometryPosToLanePos(distAfterDivergence);
                    const double lbcSibling = sibling->interpolateGeometryPosToLanePos(distAfterDivergence);
                    myLengthsBehindCrossing.push_back(std::make_pair(lbcLane, lbcSibling));
#ifdef MSLink_DEBUG_CROSSING_POINTS
                    std::cout
                            << " " << lane->getID()
                            << " merges with " << (*it_lane)->getID()
                            << " nextLane " << lane->getLinkCont()[0]->getViaLaneOrLane()->getID()
                            << " dist1=" << myLengthsBehindCrossing.back().first
                            << " dist2=" << myLengthsBehindCrossing.back().second
                            << "\n";
#endif
                }
            } else {
                std::vector<double> intersections1 = lane->getShape().intersectsAtLengths2D((*it_lane)->getShape());
#ifdef MSLink_DEBUG_CROSSING_POINTS_DETAILS
                std::cout << "    intersections1=" << toString(intersections1) << "\n";
#endif
                bool haveIntersection = true;
                if (intersections1.size() == 0) {
                    intersections1.push_back(-NO_INTERSECTION); // disregard this foe (using maxdouble leads to nasty problems down the line)
                    haveIntersection = false;
                } else if (intersections1.size() > 1) {
                    std::sort(intersections1.begin(), intersections1.end());
                }
                std::vector<double> intersections2 = (*it_lane)->getShape().intersectsAtLengths2D(lane->getShape());
#ifdef MSLink_DEBUG_CROSSING_POINTS_DETAILS
                std::cout << "    intersections2=" << toString(intersections2) << "\n";
#endif
                if (intersections2.size() == 0) {
                    intersections2.push_back(0);
                } else if (intersections2.size() > 1) {
                    std::sort(intersections2.begin(), intersections2.end());
                }
                if (haveIntersection) {
                    // lane width affects the crossing point
                    intersections1.back() -= (*it_lane)->getWidth() / 2;
                    intersections2.back() -= lane->getWidth() / 2;
                    // ensure negative offset for weird geometries
                    intersections1.back() = MAX2(0.0, intersections1.back());
                    intersections2.back() = MAX2(0.0, intersections2.back());

                    // also length/geometry factor. (XXX: Why subtract width/2 *before* converting geometric position to lane pos? refs #3031)
                    intersections1.back() = lane->interpolateGeometryPosToLanePos(intersections1.back());
                    intersections2.back() = (*it_lane)->interpolateGeometryPosToLanePos(intersections2.back());

                    if (internalLaneBefore->getLogicalPredecessorLane()->getEdge().isInternal() && !(*it_lane)->getEdge().isCrossing())  {
                        // wait at the internal junction
                        // (except for foes that are crossings since there is no internal junction)
                        intersections1.back() = 0;
                    }
                }

                myLengthsBehindCrossing.push_back(std::make_pair(
                                                      lane->getLength() - intersections1.back(),
                                                      (*it_lane)->getLength() - intersections2.back()));

#ifdef MSLink_DEBUG_CROSSING_POINTS
                std::cout
                        << "  intersection of " << lane->getID()
                        << " totalLength=" << lane->getLength()
                        << " with " << (*it_lane)->getID()
                        << " totalLength=" << (*it_lane)->getLength()
                        << " dist1=" << myLengthsBehindCrossing.back().first
                        << " dist2=" << myLengthsBehindCrossing.back().second
                        << "\n";
#endif
            }
        }
        // check for overlap with internal lanes from the same source lane
        const MSLane* pred = lane->getLogicalPredecessorLane();
        // to avoid overlap with vehicles that came from pred (especially when pred has endOffset > 0)
        // we add all other internal lanes from pred as foeLanes
        for (const MSLink* const it : pred->getLinkCont()) {
            const MSLane* sibling = it->getViaLane();
            if (sibling != lane && sibling != nullptr) {
                const double minDist = MIN2(DIVERGENCE_MIN_WIDTH, 0.5 * (lane->getWidth() + sibling->getWidth()));
                if (lane->getShape().front().distanceTo2D(sibling->getShape().front()) >= minDist) {
                    // account for lateral shift by the entry links
                    continue;
                }
                const double distToDivergence = computeDistToDivergence(lane, sibling, minDist, true);
                const double lbcLane = MAX2(0.0, lane->getLength() - lane->interpolateGeometryPosToLanePos(distToDivergence));
                const double lbcSibling = MAX2(0.0, sibling->getLength() - sibling->interpolateGeometryPosToLanePos(distToDivergence));
                myLengthsBehindCrossing.push_back(std::make_pair(lbcLane, lbcSibling));
                myFoeLanes.push_back(sibling);
#ifdef MSLink_DEBUG_CROSSING_POINTS
                std::cout << " adding same-origin foe" << sibling->getID()
                          << " dist1=" << myLengthsBehindCrossing.back().first
                          << " dist2=" << myLengthsBehindCrossing.back().second
                          << "\n";
#endif
            }
        }
    }
    if (MSGlobals::gLateralResolution > 0) {
        // check for links with the same origin lane and the same destination edge
        const MSEdge* myTarget = &myLane->getEdge();
        // save foes for entry links
        for (MSLink* const it : myLaneBefore->getLinkCont()) {
            const MSEdge* target = &(it->getLane()->getEdge());
            if (it == this) {
                continue;
            }
            if (target == myTarget) {
                mySublaneFoeLinks.push_back(it);
#ifdef MSLink_DEBUG_CROSSING_POINTS
                std::cout << "  sublaneFoeLink (same target): " << it->getViaLaneOrLane()->getID() << "\n";
#endif
            } else if (myDirection != LinkDirection::STRAIGHT && it->getDirection() == LinkDirection::STRAIGHT) {
                // potential turn conflicht
                mySublaneFoeLinks2.push_back(it);
#ifdef MSLink_DEBUG_CROSSING_POINTS
                std::cout << "  sublaneFoeLink2 (other target: " << it->getViaLaneOrLane()->getID() << "\n";
#endif
            }
        }
        // save foes for exit links
        if (fromInternalLane()) {
            //std::cout << " setRequestInformation link=" << getViaLaneOrLane()->getID() << " before=" << myLaneBefore->getID() << " before2=" << myLaneBefore->getIncomingLanes().front().lane->getID() << "\n";
            for (const MSLink* const link : myLaneBefore->getIncomingLanes().front().lane->getLinkCont()) {
                if (link->getViaLane() != myInternalLaneBefore && &link->getLane()->getEdge() == myTarget) {
                    //std::cout << " add sublaneFoe=" << (*it)->getViaLane()->getID() << "\n";
                    mySublaneFoeLanes.push_back(link->getViaLane());
                }
            }
        }
    }
    if (myInternalLaneBefore != nullptr
            && myDirection != LinkDirection::STRAIGHT
            // for right turns, the curvature helps rather than restricts the linkLeader check
            && (
                (!MSGlobals::gLefthand && myDirection != LinkDirection::RIGHT)
                || (MSGlobals::gLefthand && myDirection != LinkDirection::LEFT))) {
        const double angle = fabs(GeomHelper::angleDiff(
                                      myLaneBefore->getNormalPredecessorLane()->getShape().angleAt2D(-2),
                                      myLane->getShape().angleAt2D(0)));
        if (angle > 0) {
            double length = myInternalLaneBefore->getShape().length2D();
            if (myInternalLaneBefore->getIncomingLanes().size() == 1 &&
                    myInternalLaneBefore->getIncomingLanes()[0].lane->isInternal()) {
                length += myInternalLaneBefore->getIncomingLanes()[0].lane->getShape().length2D();
            } else if (myInternalLane != nullptr) {
                length += myInternalLane->getShape().length2D();
            }
            myRadius = length / angle;
            //std::cout << getDescription() << " a=" << RAD2DEG(angle) << " l=" << length << " r=" << myRadius << "\n";
        }
    }
}


double
MSLink::computeDistToDivergence(const MSLane* lane, const MSLane* sibling, double minDist, bool sameSource) const {
    double lbcSibling = 0;
    double lbcLane = 0;

    PositionVector l = lane->getShape();
    PositionVector s = sibling->getShape();
    double length = l.length2D();
    double sibLength = s.length2D();
    if (!sameSource) {
        l = l.reverse();
        s = s.reverse();
    } else if (sibling->getEntryLink()->myAmIndirect) {
        // ignore final waiting position since it may be quite close to the lane
        // shape but the waiting position is perpendicular (so the minDist
        // requirement is not necessary
        lbcSibling += s[-1].distanceTo2D(s[-2]);
        s.pop_back();
    } else if (lane->getEntryLink()->myAmIndirect) {
        // ignore final waiting position since it may be quite close to the lane
        // shape but the waiting position is perpendicular (so the minDist
        // requirement is not necessary
        lbcLane += l[-1].distanceTo2D(l[-2]);
        l.pop_back();
    }

#ifdef MSLink_DEBUG_CROSSING_POINTS_DETAILS
    std::cout << "   sameSource=" << sameSource << " minDist=" << minDist << " backDist=" << l.back().distanceTo2D(s.back()) << "\n";
#endif
    if (l.back().distanceTo2D(s.back()) > minDist) {
        // compute the final divergence point
        // this position serves two purposes:
        // 1) once the foe vehicle back (on sibling) has passed this point, we can safely ignore it
        // 2) both vehicles are put into a cf-relationship while before the point.
        //    Since the actual crossing point is at the start of the junction,
        //    we want to make sure that both vehicles have the same distance to the crossing point and thus follow each other naturally
        std::vector<double> distances = l.distances(s);
#ifdef MSLink_DEBUG_CROSSING_POINTS
        std::cout << "   distances=" << toString(distances) << "\n";
#endif
        assert(distances.size() == l.size() + s.size());
        if (distances.back() > minDist && distances[l.size() - 1] > minDist) {
            // do a pairwise check between lane and sibling to make because we do not know which of them bends more
            for (int j = (int)s.size() - 2; j >= 0; j--) {
                const int i = j + (int)l.size();
                const double segLength = s[j].distanceTo2D(s[j + 1]);
                if (distances[i] > minDist) {
                    lbcSibling += segLength;
                } else {
                    // assume no sharp bends and just interpolate the last segment
                    lbcSibling += segLength - (minDist - distances[i]) * segLength / (distances[i + 1] - distances[i]);
                    break;
                }
            }
            for (int i = (int)l.size() - 2; i >= 0; i--) {
                const double segLength = l[i].distanceTo2D(l[i + 1]);
                if (distances[i] > minDist) {
                    lbcLane += segLength;
                } else {
                    // assume no sharp bends and just interpolate the last segment
                    lbcLane += segLength - (minDist - distances[i]) * segLength / (distances[i + 1] - distances[i]);
                    break;
                }
            }
        }
        assert(lbcSibling >= -NUMERICAL_EPS);
        assert(lbcLane >= -NUMERICAL_EPS);
    }
    const double distToDivergence1 = sibling->getLength() - lbcSibling;
    const double distToDivergence2 = lane->getLength() - lbcLane;
    const double distToDivergence = MIN3(
                                        MAX2(distToDivergence1, distToDivergence2),
                                        sibLength, length);
#ifdef MSLink_DEBUG_CROSSING_POINTS
    std::cout << "   distToDivergence=" << distToDivergence
              << " distTD1=" << distToDivergence1
              << " distTD2=" << distToDivergence2
              << " length=" << length
              << " sibLength=" << sibLength
              << "\n";
#endif
    return distToDivergence;
}


bool
MSLink::contIntersect(const MSLane* lane, const MSLane* foe) {
    if (foe->getLinkCont()[0]->getViaLane() != nullptr) {
        std::vector<double> intersections = lane->getShape().intersectsAtLengths2D(foe->getShape());
        return intersections.size() > 0;
    }
    return false;
}


void
MSLink::setApproaching(const SUMOVehicle* approaching, const SUMOTime arrivalTime, const double arrivalSpeed, const double leaveSpeed,
                       const bool setRequest, const double arrivalSpeedBraking, const SUMOTime waitingTime, double dist, double latOffset) {
    const SUMOTime leaveTime = getLeaveTime(arrivalTime, arrivalSpeed, leaveSpeed, approaching->getVehicleType().getLength());
#ifdef DEBUG_APPROACHING
    if (DEBUG_COND2(approaching)) {
        std::cout << SIMTIME << " Link '" << (myLaneBefore == 0 ? "NULL" : myLaneBefore->getID()) << "'->'" << (myLane == 0 ? "NULL" : myLane->getID()) << "' Adding approaching vehicle '" << approaching->getID() << "'\nCurrently registered vehicles:" << std::endl;
        for (auto i = myApproachingVehicles.begin(); i != myApproachingVehicles.end(); ++i) {
            std::cout << "'" << i->first->getID() << "'" << std::endl;
        }
    }
#endif
    myApproachingVehicles.emplace(approaching,
                                  ApproachingVehicleInformation(arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, setRequest,
                                          arrivalSpeedBraking, waitingTime, dist, approaching->getSpeed(), latOffset));
}


void
MSLink::setApproaching(const SUMOVehicle* approaching, ApproachingVehicleInformation ai) {

#ifdef DEBUG_APPROACHING
    if (DEBUG_COND2(approaching)) {
        std::cout << SIMTIME << " Link '" << (myLaneBefore == 0 ? "NULL" : myLaneBefore->getID()) << "'->'" << (myLane == 0 ? "NULL" : myLane->getID()) << "' Adding approaching vehicle '" << approaching->getID() << "'\nCurrently registered vehicles:" << std::endl;
        for (auto i = myApproachingVehicles.begin(); i != myApproachingVehicles.end(); ++i) {
            std::cout << "'" << i->first->getID() << "'" << std::endl;
        }
    }
#endif
    myApproachingVehicles.emplace(approaching, ai);
}


void
MSLink::addBlockedLink(MSLink* link) {
    myBlockedFoeLinks.insert(link);
}



bool
MSLink::willHaveBlockedFoe() const {
    for (std::set<MSLink*>::const_iterator i = myBlockedFoeLinks.begin(); i != myBlockedFoeLinks.end(); ++i) {
        if ((*i)->isBlockingAnyone()) {
            return true;
        }
    }
    return false;
}


void
MSLink::removeApproaching(const SUMOVehicle* veh) {

#ifdef DEBUG_APPROACHING
    if (DEBUG_COND2(veh)) {
        std::cout << SIMTIME << " Link '" << (myLaneBefore == 0 ? "NULL" : myLaneBefore->getID()) << "'->'" << (myLane == 0 ? "NULL" : myLane->getID()) << std::endl;
        std::cout << "' Removing approaching vehicle '" << veh->getID() << "'\nCurrently registered vehicles:" << std::endl;
        for (auto i = myApproachingVehicles.begin(); i != myApproachingVehicles.end(); ++i) {
            std::cout << "'" << i->first->getID() << "'" << std::endl;
        }
    }
#endif
    myApproachingVehicles.erase(veh);
}


MSLink::ApproachingVehicleInformation
MSLink::getApproaching(const SUMOVehicle* veh) const {
    auto i = myApproachingVehicles.find(veh);
    if (i != myApproachingVehicles.end()) {
        return i->second;
    } else {
        return ApproachingVehicleInformation(INVALID_TIME, INVALID_TIME, 0, 0, false, 0, 0, 0, 0, 0);
    }
}

void
MSLink::clearState() {
    myApproachingVehicles.clear();
}

SUMOTime
MSLink::getLeaveTime(const SUMOTime arrivalTime, const double arrivalSpeed,
                     const double leaveSpeed, const double vehicleLength) const {
    return arrivalTime + TIME2STEPS((getLength() + vehicleLength) / MAX2(0.5 * (arrivalSpeed + leaveSpeed), NUMERICAL_EPS));
}


bool
MSLink::opened(SUMOTime arrivalTime, double arrivalSpeed, double leaveSpeed, double vehicleLength,
               double impatience, double decel, SUMOTime waitingTime, double posLat,
               BlockingFoes* collectFoes, bool ignoreRed, const SUMOTrafficObject* ego) const {
#ifdef MSLink_DEBUG_OPENED
    if (gDebugFlag1) {
        std::cout << SIMTIME << "  opened? link=" << getDescription() << " red=" << haveRed() << " cont=" << isCont() << " numFoeLinks=" << myFoeLinks.size() << " havePrio=" << havePriority() << " lastWasContMajorGreen=" << lastWasContState(LINKSTATE_TL_GREEN_MAJOR) << "\n";
    }
#endif
    if (haveRed() && !ignoreRed) {
        return false;
    }
    if (isCont() && MSGlobals::gUsingInternalLanes) {
        return true;
    }
    const SUMOTime leaveTime = getLeaveTime(arrivalTime, arrivalSpeed, leaveSpeed, vehicleLength);
    if (MSGlobals::gLateralResolution > 0) {
        // check for foes on the same lane with the same target edge
        for (const MSLink* foeLink : mySublaneFoeLinks) {
            assert(myLane != foeLink->getLane());
            for (const auto& it : foeLink->myApproachingVehicles) {
                const SUMOVehicle* foe = it.first;
                if (
                    // there only is a conflict if the paths cross
                    ((posLat < foe->getLateralPositionOnLane() + it.second.latOffset && myLane->getIndex() > foeLink->myLane->getIndex())
                     || (posLat > foe->getLateralPositionOnLane() + it.second.latOffset && myLane->getIndex() < foeLink->myLane->getIndex()))
                    // the vehicle that arrives later must yield
                    && (arrivalTime > it.second.arrivalTime
                        // if both vehicles arrive at the same time, the one
                        // to the left must yield
                        || (arrivalTime == it.second.arrivalTime && posLat > foe->getLateralPositionOnLane()))) {
                    if (blockedByFoe(foe, it.second, arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, false,
                                     impatience, decel, waitingTime, ego)) {
#ifdef MSLink_DEBUG_OPENED
                        if (gDebugFlag1) {
                            std::cout << SIMTIME << " blocked by " << foe->getID() << " arrival=" << arrivalTime << " foeArrival=" << it.second.arrivalTime << "\n";
                        }
#endif
                        if (collectFoes == nullptr) {
#ifdef MSLink_DEBUG_OPENED
                            if (gDebugFlag1) {
                                std::cout << " link=" << getViaLaneOrLane()->getID() << " blocked by sublaneFoe=" << foe->getID() << " foeLink=" << foeLink->getViaLaneOrLane()->getID() << " posLat=" << posLat << "\n";
                            }
#endif
                            return false;
                        } else {
                            collectFoes->push_back(it.first);
                        }
                    }
                }
            }
        }
        // check for foes on the same lane with a different target edge
        // (straight movers take precedence if the paths cross)
        const int lhSign = MSGlobals::gLefthand ? -1 : 1;
        for (const MSLink* foeLink : mySublaneFoeLinks2) {
            assert(myDirection != LinkDirection::STRAIGHT);
            for (const auto& it : foeLink->myApproachingVehicles) {
                const SUMOVehicle* foe = it.first;
                // there only is a conflict if the paths cross
                // and if the vehicles are not currently in a car-following relationship
                const double egoWidth = ego == nullptr ? 1.8 : ego->getVehicleType().getWidth();
                if (!lateralOverlap(posLat, egoWidth, foe->getLateralPositionOnLane() + it.second.latOffset, foe->getVehicleType().getWidth())
                        && (((myDirection == LinkDirection::RIGHT || myDirection == LinkDirection::PARTRIGHT)
                             && (posLat * lhSign > (foe->getLateralPositionOnLane() + it.second.latOffset) * lhSign))
                            || ((myDirection == LinkDirection::LEFT || myDirection == LinkDirection::PARTLEFT)
                                && (posLat * lhSign < (foe->getLateralPositionOnLane() + it.second.latOffset) * lhSign)))) {
                    if (blockedByFoe(foe, it.second, arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, false,
                                     impatience, decel, waitingTime, ego)) {
#ifdef MSLink_DEBUG_OPENED
                        if (gDebugFlag1) {
                            std::cout << SIMTIME << " blocked by sublane foe " << foe->getID() << " arrival=" << arrivalTime << " foeArrival=" << it.second.arrivalTime << "\n";
                        }
#endif
                        if (collectFoes == nullptr) {
#ifdef MSLink_DEBUG_OPENED
                            if (gDebugFlag1) {
                                std::cout << " link=" << getViaLaneOrLane()->getID() << " blocked by sublaneFoe2=" << foe->getID() << " foeLink=" << foeLink->getViaLaneOrLane()->getID() << " posLat=" << posLat << "\n";
                            }
#endif
                            return false;
                        } else {
                            collectFoes->push_back(it.first);
                        }
                    }
                }
            }
        }
    }
    if ((havePriority() || lastWasContState(LINKSTATE_TL_GREEN_MAJOR)) && myState != LINKSTATE_ZIPPER) {
        // priority usually means the link is open but there are exceptions:
        // zipper still needs to collect foes
        // sublane model could have detected a conflict
        return collectFoes == nullptr || collectFoes->size() == 0;
    }
    if ((myState == LINKSTATE_STOP || myState == LINKSTATE_ALLWAY_STOP) && waitingTime == 0) {
        return false;
    }

#ifdef MSLink_DEBUG_OPENED
    if (gDebugFlag1) {
        std::cout << SIMTIME << " opened link=" << getViaLaneOrLane()->getID() << " foeLinks=" << myFoeLinks.size() << "\n";
    }
#endif

    if (MSGlobals::gUseMesoSim && impatience == 1) {
        return true;
    }
    const bool lastWasContRed = lastWasContState(LINKSTATE_TL_RED);
    for (const MSLink* const link : myFoeLinks) {
        if (MSGlobals::gUseMesoSim) {
            if (link->haveRed()) {
                continue;
            }
        }
#ifdef MSLink_DEBUG_OPENED
        if (gDebugFlag1) {
            std::cout << "    foeLink=" << link->getViaLaneOrLane()->getID() << " numApproaching=" << link->getApproaching().size() << "\n";
        }
#endif
        if (link->blockedAtTime(arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, myLane == link->getLane(),
                                impatience, decel, waitingTime, collectFoes, ego, lastWasContRed)) {
            return false;
        }
    }
    if (collectFoes != nullptr && collectFoes->size() > 0) {
        return false;
    }
    return true;
}


bool
MSLink::blockedAtTime(SUMOTime arrivalTime, SUMOTime leaveTime, double arrivalSpeed, double leaveSpeed,
                      bool sameTargetLane, double impatience, double decel, SUMOTime waitingTime,
                      BlockingFoes* collectFoes, const SUMOTrafficObject* ego, bool lastWasContRed) const {
    for (const auto& it : myApproachingVehicles) {
#ifdef MSLink_DEBUG_OPENED
        if (gDebugFlag1) {
            if (ego != nullptr
                    && ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_SPEED, 0) >= it.second.speed
                    && ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_PROB, 0) > 0) {
                std::stringstream stream; // to reduce output interleaving from different threads
                stream << SIMTIME << " " << myApproachingVehicles.size() << "   foe link=" << getViaLaneOrLane()->getID()
                       << " foeVeh=" << it.first->getID() << " (below ignore speed)"
                       << " ignoreFoeProb=" << ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_PROB, 0)
                       << "\n";
                std::cout << stream.str();
            }
        }
#endif
        if (it.first != ego
                && (ego == nullptr
                    || ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_PROB, 0) == 0
                    || ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_SPEED, 0) < it.second.speed
                    || ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_IGNORE_FOE_PROB, 0) < RandHelper::rand(ego->getRNG()))
                && !ignoreFoe(ego, it.first)
                && (!lastWasContRed || it.first->getSpeed() > SUMO_const_haltingSpeed)
                && blockedByFoe(it.first, it.second, arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, sameTargetLane,
                                impatience, decel, waitingTime, ego)) {
            if (collectFoes == nullptr) {
                return true;
            } else {
                collectFoes->push_back(it.first);
            }
        }
    }
    return false;
}


bool
MSLink::blockedByFoe(const SUMOVehicle* veh, const ApproachingVehicleInformation& avi,
                     SUMOTime arrivalTime, SUMOTime leaveTime, double arrivalSpeed, double leaveSpeed,
                     bool sameTargetLane, double impatience, double decel, SUMOTime waitingTime,
                     const SUMOTrafficObject* ego) const {
#ifdef MSLink_DEBUG_OPENED
    if (gDebugFlag1) {
        std::stringstream stream; // to reduce output interleaving from different threads
        stream << "    link=" << getDescription()
               << " foeVeh=" << veh->getID()
               << " req=" << avi.willPass
               << " aT=" << avi.arrivalTime
               << " lT=" << avi.leavingTime
               << "\n";
        std::cout << stream.str();
    }
#endif
    if (!avi.willPass) {
        return false;
    }
    if (myState == LINKSTATE_ALLWAY_STOP) {
        assert(waitingTime > 0);
        if (waitingTime > avi.waitingTime) {
            return false;
        }
        if (waitingTime == avi.waitingTime && arrivalTime < avi.arrivalTime) {
            return false;
        }
    }
    SUMOTime foeArrivalTime = avi.arrivalTime;
    double foeArrivalSpeedBraking = avi.arrivalSpeedBraking;
    if (impatience > 0 && arrivalTime < avi.arrivalTime) {
#ifdef MSLink_DEBUG_OPENED
        gDebugFlag6 = ((ego == nullptr || ego->isSelected()) && (veh == nullptr || veh->isSelected()));
#endif
        const SUMOTime fatb = computeFoeArrivalTimeBraking(arrivalTime, veh, avi.arrivalTime, impatience, avi.dist, foeArrivalSpeedBraking);
        foeArrivalTime = (SUMOTime)((1. - impatience) * (double)avi.arrivalTime + impatience * (double)fatb);
#ifdef MSLink_DEBUG_OPENED
        if (gDebugFlag6) {
            std::cout << SIMTIME << " link=" << getDescription() << " ego=" << ego->getID() << " foe=" << veh->getID()
                      << " at=" << STEPS2TIME(arrivalTime)
                      << " fat=" << STEPS2TIME(avi.arrivalTime)
                      << " fatb=" << STEPS2TIME(fatb)
                      << " fat2=" << STEPS2TIME(foeArrivalTime)
                      << "\n";
        }
#endif
    }


    const SUMOTime lookAhead = (myState == LINKSTATE_ZIPPER
                                ? myLookaheadTimeZipper
                                : (ego == nullptr
                                   ? myLookaheadTime
                                   : TIME2STEPS(ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_TIMEGAP_MINOR, STEPS2TIME(myLookaheadTime)))));
    //if (ego != 0) std::cout << SIMTIME << " ego=" << ego->getID() << " jmTimegapMinor=" << ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_TIMEGAP_MINOR, -1) << " lookAhead=" << lookAhead << "\n";
#ifdef MSLink_DEBUG_OPENED
    if (gDebugFlag1 || gDebugFlag6) {
        std::stringstream stream; // to reduce output interleaving from different threads
        stream << "       imp=" << impatience << " fAT2=" << foeArrivalTime << " fASb=" << foeArrivalSpeedBraking << " lA=" << lookAhead << " egoAT=" << arrivalTime << " egoLT=" << leaveTime << " egoLS=" << leaveSpeed << "\n";
        std::cout << stream.str();
    }
#endif
    if (avi.leavingTime < arrivalTime) {
        // ego wants to be follower
        if (sameTargetLane && (arrivalTime - avi.leavingTime < lookAhead
                               || unsafeMergeSpeeds(avi.leaveSpeed, arrivalSpeed,
                                       veh->getVehicleType().getCarFollowModel().getMaxDecel(), decel))) {
#ifdef MSLink_DEBUG_OPENED
            if (gDebugFlag1 || gDebugFlag6) {
                std::cout << "      blocked (cannot follow)\n";
            }
#endif
            return true;
        }
    } else if (foeArrivalTime > leaveTime + lookAhead) {
        // ego wants to be leader.
        if (sameTargetLane && unsafeMergeSpeeds(leaveSpeed, foeArrivalSpeedBraking,
                                                decel, veh->getVehicleType().getCarFollowModel().getMaxDecel())) {
#ifdef MSLink_DEBUG_OPENED
            if (gDebugFlag1 || gDebugFlag6) {
                std::cout << "      blocked (cannot lead)\n";
            }
#endif
            return true;
        }
    } else {
        // even without considering safeHeadwayTime there is already a conflict
#ifdef MSLink_DEBUG_OPENED
        if (gDebugFlag1 || gDebugFlag6) {
            std::cout << "      blocked (hard conflict)\n";
        }
#endif
        return true;
    }
    return false;
}


SUMOTime
MSLink::computeFoeArrivalTimeBraking(SUMOTime arrivalTime, const SUMOVehicle* foe, SUMOTime foeArrivalTime, double impatience, double dist, double& fasb) {
    // a: distance saved when foe brakes from arrivalTime to foeArrivalTime
    // b: distance driven past foeArrivalTime
    // m: permitted decceleration
    // d: total deceleration until foeArrivalTime
    // dist2: distance of foe at arrivalTime
    // actual arrivalTime must fall on a simulation step
    if (arrivalTime - arrivalTime % DELTA_T == foeArrivalTime - foeArrivalTime % DELTA_T) {
        // foe enters the junction in the same step
        return foeArrivalTime;
    }
    //arrivalTime += DELTA_T - arrivalTime % DELTA_T;
    const double m = foe->getVehicleType().getCarFollowModel().getMaxDecel() * impatience;
    const double dt = STEPS2TIME(foeArrivalTime - arrivalTime);
    const double d = dt * m;
    const double a = dt * d / 2;
    const double v = dist / STEPS2TIME(foeArrivalTime - SIMSTEP + DELTA_T);
    const double dist2 = dist - v * STEPS2TIME(arrivalTime - SIMSTEP);
    if (0.5 * v * v / m <= dist2) {
        if (gDebugFlag6) {
            std::cout << "   dist=" << dist << " dist2=" << dist2 << " at=" << STEPS2TIME(arrivalTime) << " m=" << m << " d=" << d << " a=" << a << " canBrakeToStop\n";
        }
        fasb = 0;
        return foeArrivalTime + TIME2STEPS(30);
    }
    // a = b (foe reaches the original distance to the stop line)
    // x: time driven past foeArrivalTime
    // v: foe speed without braking
    // v2: average foe speed after foeArrivalTime (braking continues for time x)
    // v2 = (v - d - x * m / 2)
    // b = v2 * x
    // solving for x gives:
    const double x = (sqrt(4 * (v - d) * (v - d) - 8 * m * a) * -0.5 - d + v) / m;

#ifdef MSLink_DEBUG_OPENED
    const double x2 = (sqrt(4 * (v - d) * (v - d) - 8 * m * a) * 0.5 - d + v) / m;
    if (gDebugFlag6 || ISNAN(x)) {
        std::cout << SIMTIME << "   dist=" << dist << " dist2=" << dist2  << " at=" << STEPS2TIME(arrivalTime) << " m=" << m << " d=" << d << " v=" << v << " a=" << a << " x=" << x << " x2=" << x2 << "\n";
    }
#endif
    fasb = v - (dt + x) * m;
    return foeArrivalTime + TIME2STEPS(x);
}


bool
MSLink::hasApproachingFoe(SUMOTime arrivalTime, SUMOTime leaveTime, double speed, double decel) const {
    for (const MSLink* const link : myFoeLinks) {
        if (link->blockedAtTime(arrivalTime, leaveTime, speed, speed, myLane == link->getLane(), 0, decel, 0)) {
            return true;
        }
    }
    for (const MSLane* const lane : myFoeLanes) {
        if (lane->getVehicleNumberWithPartials() > 0) {
            return true;
        }
    }
    return false;
}


std::pair<const SUMOVehicle*, const MSLink*>
MSLink::getFirstApproachingFoe(const MSLink* wrapAround) const {
    double closetDist = std::numeric_limits<double>::max();
    const SUMOVehicle* closest = nullptr;
    const MSLink* foeLink = nullptr;
    for (MSLink* link : myFoeLinks) {
        for (const auto& it : link->myApproachingVehicles) {
            //std::cout << " link=" << getDescription() << " foeLink_in=" << link->getLaneBefore()->getID() << " wrapAround=" << wrapAround->getDescription() << "\n";
            if (link->getLaneBefore() == wrapAround->getLaneBefore()) {
                return std::make_pair(nullptr, wrapAround);
            } else if (it.second.dist < closetDist) {
                closetDist = it.second.dist;
                if (it.second.willPass) {
                    closest = it.first;
                    foeLink = link;
                }
            }
        }
    }
    return std::make_pair(closest, foeLink);
}


void
MSLink::setTLState(LinkState state, SUMOTime t) {
    if (myState != state) {
        myLastStateChange = t;
    }
    myState = state;
    if (haveGreen()) {
        myLastGreenState = myState;
    }
}


bool
MSLink::isCont() const {
    // when a traffic light is switched off minor roads have their cont status revoked
    return myState != LINKSTATE_TL_OFF_BLINKING ? myAmCont : myAmContOff;
}


bool
MSLink::lastWasContMajor() const {
    if (myInternalLane == nullptr || myAmCont) {
        return false;
    } else {
        MSLane* pred = myInternalLane->getLogicalPredecessorLane();
        if (!pred->getEdge().isInternal()) {
            return false;
        } else {
            const MSLane* const pred2 = pred->getLogicalPredecessorLane();
            assert(pred2 != nullptr);
            const MSLink* const predLink = pred2->getLinkTo(pred);
            assert(predLink != nullptr);
            if (predLink->havePriority()) {
                return true;
            }
            if (myHavePedestrianCrossingFoe) {
                return predLink->getLastGreenState() == LINKSTATE_TL_GREEN_MAJOR;
            } else {
                return predLink->haveYellow();
            }
        }
    }
}


bool
MSLink::lastWasContState(LinkState linkState) const {
    if (myInternalLane == nullptr || myAmCont || myHavePedestrianCrossingFoe) {
        return false;
    } else {
        MSLane* pred = myInternalLane->getLogicalPredecessorLane();
        if (!pred->getEdge().isInternal()) {
            return false;
        } else {
            const MSLane* const pred2 = pred->getLogicalPredecessorLane();
            assert(pred2 != nullptr);
            const MSLink* const predLink = pred2->getLinkTo(pred);
            assert(predLink != nullptr);
            return predLink->getState() == linkState;
        }
    }
}


void
MSLink::writeApproaching(OutputDevice& od, const std::string fromLaneID) const {
    if (myApproachingVehicles.size() > 0) {
        od.openTag("link");
        od.writeAttr(SUMO_ATTR_FROM, fromLaneID);
        const std::string via = getViaLane() == nullptr ? "" : getViaLane()->getID();
        od.writeAttr(SUMO_ATTR_VIA, via);
        od.writeAttr(SUMO_ATTR_TO, getLane() == nullptr ? "" : getLane()->getID());
        std::vector<std::pair<SUMOTime, const SUMOVehicle*> > toSort; // stabilize output
        for (auto it : myApproachingVehicles) {
            toSort.push_back(std::make_pair(it.second.arrivalTime, it.first));
        }
        std::sort(toSort.begin(), toSort.end());
        for (std::vector<std::pair<SUMOTime, const SUMOVehicle*> >::const_iterator it = toSort.begin(); it != toSort.end(); ++it) {
            od.openTag("approaching");
            const ApproachingVehicleInformation& avi = myApproachingVehicles.find(it->second)->second;
            od.writeAttr(SUMO_ATTR_ID, it->second->getID());
            od.writeAttr(SUMO_ATTR_IMPATIENCE, it->second->getImpatience());
            od.writeAttr("arrivalTime", time2string(avi.arrivalTime));
            od.writeAttr("leaveTime", time2string(avi.leavingTime));
            od.writeAttr("arrivalSpeed", toString(avi.arrivalSpeed));
            od.writeAttr("arrivalSpeedBraking", toString(avi.arrivalSpeedBraking));
            od.writeAttr("leaveSpeed", toString(avi.leaveSpeed));
            od.writeAttr("willPass", toString(avi.willPass));
            od.closeTag();
        }
        od.closeTag();
    }
}


double
MSLink::getInternalLengthsAfter() const {
    double len = 0.;
    MSLane* lane = myInternalLane;

    while (lane != nullptr && lane->isInternal()) {
        len += lane->getLength();
        lane = lane->getLinkCont()[0]->getViaLane();
    }
    return len;
}

double
MSLink::getInternalLengthsBefore() const {
    double len = 0.;
    const MSLane* lane = myInternalLane;

    while (lane != nullptr && lane->isInternal()) {
        len += lane->getLength();
        if (lane->getIncomingLanes().size() == 1) {
            lane = lane->getIncomingLanes()[0].lane;
        } else {
            break;
        }
    }
    return len;
}


double
MSLink::getLengthsBeforeCrossing(const MSLane* foeLane) const {
    MSLane* via = myInternalLane;
    double totalDist = 0.;
    bool foundCrossing = false;
    while (via != nullptr) {
        MSLink* link = via->getLinkCont()[0];
        double dist = link->getLengthBeforeCrossing(foeLane);
        if (dist != INVALID_DOUBLE) {
            // found conflicting lane
            totalDist += dist;
            foundCrossing = true;
            break;
        } else {
            totalDist += via->getLength();
            via = link->getViaLane();
        }
    }
    if (foundCrossing) {
        return totalDist;
    } else {
        return INVALID_DOUBLE;
    }
}


double
MSLink::getLengthBeforeCrossing(const MSLane* foeLane) const {
    int foe_ix;
    for (foe_ix = 0; foe_ix != (int)myFoeLanes.size(); ++foe_ix) {
        if (myFoeLanes[foe_ix] == foeLane) {
            break;
        }
    }
    if (foe_ix == (int)myFoeLanes.size()) {
        // no conflict with the given lane, indicate by returning -1
#ifdef MSLink_DEBUG_CROSSING_POINTS
        std::cout << "No crossing of lanes '" << foeLane->getID() << "' and '" << myInternalLaneBefore->getID() << "'" << std::endl;
#endif
        return INVALID_DOUBLE;
    } else {
        // found conflicting lane index
        double dist = myInternalLaneBefore->getLength() - myLengthsBehindCrossing[foe_ix].first;
        if (dist == -10000.) {
            // this is the value in myLengthsBehindCrossing, if the relation allows intersection but none is present for the actual geometry.
            return INVALID_DOUBLE;
        }
#ifdef MSLink_DEBUG_CROSSING_POINTS
        std::cout << "Crossing of lanes '" << myInternalLaneBefore->getID() << "' and '" << foeLane->getID()
                  << "' at distance " << dist << " (approach along '"
                  <<  myInternalLaneBefore->getEntryLink()->getLaneBefore()->getID() << "')" << std::endl;
#endif
        return dist;
    }
}


bool
MSLink::isEntryLink() const {
    if (MSGlobals::gUsingInternalLanes) {
        return myInternalLane != nullptr && myInternalLaneBefore == nullptr;
    } else {
        return false;
    }
}

bool
MSLink::isConflictEntryLink() const {
    // either a non-cont entry link or the link after a cont-link
    return !myAmCont && (isEntryLink() || (myInternalLaneBefore != nullptr && myInternalLane != nullptr));
}

bool
MSLink::isExitLink() const {
    if (MSGlobals::gUsingInternalLanes) {
        return myInternalLaneBefore != nullptr && myInternalLane == nullptr;
    } else {
        return false;
    }
}

bool
MSLink::isExitLinkAfterInternalJunction() const {
    if (MSGlobals::gUsingInternalLanes) {
        return (getInternalLaneBefore() != nullptr
                && myInternalLaneBefore->getIncomingLanes().size() == 1
                && myInternalLaneBefore->getIncomingLanes().front().viaLink->isInternalJunctionLink());
    } else {
        return false;
    }
}


const MSLink*
MSLink::getCorrespondingExitLink() const {
    MSLane* lane = myInternalLane;
    const MSLink* link = nullptr;
    while (lane != nullptr) {
        link = lane->getLinkCont()[0];
        lane = link->getViaLane();
    }
    return link;
}


const MSLink*
MSLink::getCorrespondingEntryLink() const {
    const MSLink* link = this;
    while (link->myLaneBefore->isInternal()) {
        assert(myLaneBefore->getIncomingLanes().size() == 1);
        link = link->myLaneBefore->getIncomingLanes().front().viaLink;
    }
    return link;
}


bool
MSLink::isInternalJunctionLink() const {
    return getInternalLaneBefore() != nullptr && myInternalLane != nullptr;
}


const MSLink::LinkLeaders
MSLink::getLeaderInfo(const MSVehicle* ego, double dist, std::vector<const MSPerson*>* collectBlockers, bool isShadowLink) const {
    LinkLeaders result;
    // this link needs to start at an internal lane (either an exit link or between two internal lanes)
    // or it must be queried by the pedestrian model (ego == 0)
    if (ego != nullptr && (!fromInternalLane() || ego->getLaneChangeModel().isOpposite())) {
        // ignore link leaders
        return result;
    }
    //gDebugFlag1 = true;
    if (gDebugFlag1) {
        std::cout << SIMTIME << " getLeaderInfo link=" << getViaLaneOrLane()->getID() << " dist=" << dist << " isShadowLink=" << isShadowLink << "\n";
    }
    if (MSGlobals::gComputeLC && ego != nullptr && ego->getLane()->isNormal()) {
        const MSLink* junctionEntry = getLaneBefore()->getEntryLink();
        if (junctionEntry->haveRed() && !ego->ignoreRed(junctionEntry, true)) {
            if (gDebugFlag1) {
                std::cout << "   ignore linkLeaders beyond red light\n";
            }
            return result;
        }
    }
    // this is an exit link
    for (int i = 0; i < (int)myFoeLanes.size(); ++i) {
        const MSLane* foeLane = myFoeLanes[i];
        const MSLink* foeExitLink = foeLane->getLinkCont()[0];
        // distance from the querying vehicle to the crossing point with foeLane
        double distToCrossing = dist - myLengthsBehindCrossing[i].first;
        const bool sameTarget = (myLane == foeExitLink->getLane()) && !isInternalJunctionLink();
        const bool sameSource = (myInternalLaneBefore != nullptr && myInternalLaneBefore->getLogicalPredecessorLane() == foeLane->getLogicalPredecessorLane());
        const double crossingWidth = (sameTarget || sameSource) ? 0 : foeLane->getWidth();
        const double foeCrossingWidth = (sameTarget || sameSource) ? 0 : myInternalLaneBefore->getWidth();
        // special treatment of contLane foe only applies if this lane is not a contLane or contLane follower itself
        const bool contLane = (foeExitLink->getViaLaneOrLane()->getEdge().isInternal() && !(
                                   isInternalJunctionLink() || isExitLinkAfterInternalJunction()));
        if (gDebugFlag1) {
            std::cout << " distToCrossing=" << distToCrossing << " foeLane=" << foeLane->getID() << " cWidth=" << crossingWidth
                      << " ijl=" << isInternalJunctionLink() << " sT=" << sameTarget << " sS=" << sameSource
                      << " lbc=" << myLengthsBehindCrossing[i].first
                      << " flbc=" << myLengthsBehindCrossing[i].second
                      << " contLane=" << contLane
                      << " state=" << toString(myState)
                      << " foeState=" << toString(foeExitLink->getState())
                      << "\n";
        }
        if (distToCrossing + crossingWidth < 0 && !sameTarget
                && (ego == nullptr || !MSGlobals::gComputeLC || distToCrossing + crossingWidth + ego->getVehicleType().getLength() < 0)) {
            continue; // vehicle is behind the crossing point, continue with next foe lane
        }
        bool ignoreGreenCont = false;
        bool foeIndirect = false;
        if (contLane) {
            const MSLink* entry = getLaneBefore()->getEntryLink();
            const MSLink* foeEntry = foeLane->getEntryLink();
            foeIndirect = foeEntry->myAmIndirect;
            if (entry != nullptr && entry->haveGreen()
                    && foeEntry != nullptr && foeEntry->haveGreen()
                    && entry->myLaneBefore != foeEntry->myLaneBefore)  {
                // ignore vehicles before an internaljunction as long as they are still in green minor mode
                ignoreGreenCont = true;
            }
        }
        if (foeIndirect && distToCrossing >= NO_INTERSECTION) {
            if (gDebugFlag1) {
                std::cout << " ignore:noIntersection\n";
            }
            continue;
        }
        const double foeDistToCrossing = foeLane->getLength() - myLengthsBehindCrossing[i].second;
        // it is not sufficient to return the last vehicle on the foeLane because ego might be its leader
        // therefore we return all vehicles on the lane
        //
        // special care must be taken for continuation lanes. (next lane is also internal)
        // vehicles on cont. lanes or on internal lanes with the same target as this link can not be ignored
        // and should block (gap = -1) unless they are part of an indirect turn
        MSLane::AnyVehicleIterator end = foeLane->anyVehiclesEnd();
        for (MSLane::AnyVehicleIterator it_veh = foeLane->anyVehiclesBegin(); it_veh != end; ++it_veh) {
            MSVehicle* leader = (MSVehicle*)*it_veh;
            const double leaderBack = leader->getBackPositionOnLane(foeLane);
            const double leaderBackDist = foeDistToCrossing - leaderBack;
            const double l2 = ego != nullptr ? ego->getLength() + 2 : 0; // add some slack to account for further meeting-angle effects
            const double sagitta = ego != nullptr && myRadius != std::numeric_limits<double>::max() ? myRadius - sqrt(myRadius * myRadius - 0.25 * l2 * l2) : 0;
            const bool pastTheCrossingPoint = leaderBackDist + foeCrossingWidth  + sagitta < 0;
            const bool foeIsBicycleTurn = (leader->getVehicleType().getVehicleClass() == SVC_BICYCLE
                                           && foeLane->getIncomingLanes().front().viaLink->getDirection() == LinkDirection::LEFT);
            const bool ignoreIndirectBicycleTurn = pastTheCrossingPoint && foeIsBicycleTurn;
            const bool cannotIgnore = ((contLane && !ignoreIndirectBicycleTurn) || sameTarget || sameSource) && ego != nullptr;
            const bool inTheWay = (((!pastTheCrossingPoint && distToCrossing > 0) || (sameTarget && distToCrossing > leaderBackDist - leader->getLength()))
                                   && leaderBackDist < leader->getVehicleType().getLength()
                                   && (!foeExitLink->isInternalJunctionLink() || foeIsBicycleTurn));
            const bool isOpposite = leader->getLaneChangeModel().isOpposite();
            const auto avi = foeExitLink->getApproaching(leader);
            // if leader is not found, assume that it performed a lane change in the last step
            const bool willPass = avi.willPass || (avi.arrivalTime == INVALID_TIME && sameTarget);
            if (gDebugFlag1) {
                std::cout << " candidate leader=" << leader->getID()
                          << " cannotIgnore=" << cannotIgnore
                          << " fdtc=" << foeDistToCrossing
                          << " lb=" << leaderBack
                          << " lbd=" << leaderBackDist
                          << " fcwidth=" << foeCrossingWidth
                          << " r=" << myRadius
                          << " sagitta=" << sagitta
                          << " foePastCP=" << pastTheCrossingPoint
                          << " inTheWay=" << inTheWay
                          << " willPass=" << willPass
                          << " isFrontOnLane=" << leader->isFrontOnLane(foeLane)
                          << " ignoreGreenCont=" << ignoreGreenCont
                          << " foeIndirect=" << foeIndirect
                          << " foeBikeTurn=" << foeIsBicycleTurn
                          << " isOpposite=" << isOpposite << "\n";
            }
            if (leader == ego) {
                continue;
            }
            // ignore greenCont foe vehicles that are not in the way
            if (!inTheWay && ignoreGreenCont) {
                if (gDebugFlag1) {
                    std::cout << "   ignoreGreenCont\n";
                }
                continue;
            }
            // after entering the conflict area, ignore foe vehicles that are not in the way
            if (distToCrossing < -POSITION_EPS && !inTheWay
                    && (ego == nullptr || !MSGlobals::gComputeLC || distToCrossing < -ego->getVehicleType().getLength())) {
                if (gDebugFlag1) {
                    std::cout << "   ego entered conflict area\n";
                }
                continue;
            }
            // ignore foe vehicles that will not pass
            if ((!cannotIgnore || leader->isStopped() || sameTarget)
                    && !willPass
                    && leader->isFrontOnLane(foeLane)
                    && !isOpposite
                    && !inTheWay
                    // willPass is false if the vehicle is already on the stopping edge
                    && !leader->willStop()) {
                if (gDebugFlag1) {
                    std::cout << "   foe will not pass\n";
                }
                continue;
            }
            // check whether foe is blocked and might need to change before leaving the junction
            const bool foeStrategicBlocked = (leader->getLaneChangeModel().isStrategicBlocked() &&
                                              leader->getCarFollowModel().brakeGap(leader->getSpeed()) <= foeLane->getLength() - leaderBack);

            if (MSGlobals::gSublane && ego != nullptr && (sameSource || sameTarget)
                    && (!foeStrategicBlocked)) {
                if (ego->getLane() == leader->getLane()) {
                    continue;
                }
                // ignore vehicles if not in conflict sublane-wise
                const double posLat = ego->getLateralPositionOnLane();
                const double posLatLeader = leader->getLateralPositionOnLane() + leader->getLatOffset(foeLane);
                const double latGap = (fabs(posLat - posLatLeader)
                                       - 0.5 * (ego->getVehicleType().getWidth() + leader->getVehicleType().getWidth()));
                const double maneuverDist = leader->getLaneChangeModel().getManeuverDist() * (posLat < posLatLeader ? -1 : 1);
                if (gDebugFlag1) {
                    std::cout << " sublaneFoe lane=" << myInternalLaneBefore->getID()
                              << " sameSource=" << sameSource
                              << " sameTarget=" << sameTarget
                              << " foeLane=" << foeLane->getID()
                              << " leader=" << leader->getID()
                              << " egoLane=" << ego->getLane()->getID()
                              << " leaderLane=" << leader->getLane()->getID()
                              << " egoLat=" << posLat
                              << " leaderLat=" << posLatLeader
                              << " leaderLatOffset=" << leader->getLatOffset(foeLane)
                              << " latGap=" << latGap
                              << " maneuverDist=" << maneuverDist
                              << " computeLC=" << MSGlobals::gComputeLC
                              << " egoMaxSpeedLat=" << ego->getVehicleType().getMaxSpeedLat()
                              << "\n";
                }
                if (latGap > 0 && (latGap > maneuverDist || !sameTarget)
                        // do not perform sublane changes that interfere with the leader vehicle
                        && (!MSGlobals::gComputeLC || latGap > ego->getVehicleType().getMaxSpeedLat())) {
                    const MSLink* foeEntryLink = foeLane->getIncomingLanes().front().viaLink;
                    if (sameSource) {
                        // for lanes from the same edge, higer index implies a
                        // connection further to the left
                        const bool leaderFromRight = (myIndex > foeEntryLink->getIndex());
                        if ((posLat > posLatLeader) == leaderFromRight) {
                            // ignore speed since lanes diverge
                            if (gDebugFlag1) {
                                std::cout << "   ignored (same source) leaderFromRight=" << leaderFromRight << "\n";
                            }
                            continue;
                        }
                    } else {
                        // for lanes from different edges we cannot rely on the
                        // index due to wrap-around issues
                        if (myDirection != foeEntryLink->getDirection()) {
                            bool leaderFromRight = foeEntryLink->getDirection() < myDirection;
                            // leader vehicle should not move towards ego
                            if (MSGlobals::gLefthand) {
                                leaderFromRight = !leaderFromRight;
                            }
                            if ((posLat > posLatLeader) == leaderFromRight
                                    // leader should keep lateral position or move away from ego
                                    && (leader->getLaneChangeModel().getSpeedLat() == 0
                                        || leaderFromRight == (leader->getLaneChangeModel().getSpeedLat() < latGap))
                                    && (ego->getLaneChangeModel().getSpeedLat() == 0
                                        || leaderFromRight == (ego->getLaneChangeModel().getSpeedLat() > latGap))) {
                                if (gDebugFlag1) {
                                    std::cout << "   ignored (different source) leaderFromRight=" << leaderFromRight << "\n";
                                }
                                continue;
                            }
                        } else {
                            // XXX figure out relative direction somehow
                        }
                    }
                }
            }
            if (leader->getWaitingTime() < MSGlobals::gIgnoreJunctionBlocker) {
                // compute distance between vehicles on the the superimposition of both lanes
                // where the crossing point is the common point
                double gap;
                bool fromLeft = true;
                if (ego == nullptr) {
                    // request from pedestrian model. return distance between leaderBack and crossing point
                    //std::cout << "   foeLane=" << foeLane->getID() << " leaderBack=" << leaderBack << " foeDistToCrossing=" << foeDistToCrossing << " foeLength=" << foeLane->getLength() << " foebehind=" << myLengthsBehindCrossing[i].second << " dist=" << dist << " behind=" << myLengthsBehindCrossing[i].first << "\n";
                    gap = leaderBackDist;
                    // distToCrossing should not take into account the with of the foe lane
                    // (which was subtracted in setRequestInformation)
                    // Instead, the width of the foe vehicle is used directly by the caller.
                    distToCrossing += foeLane->getWidth() / 2;
                    if (gap + foeCrossingWidth < 0) {
                        // leader is completely past the crossing point
                        // or there is no crossing point
                        continue; // next vehicle
                    }
                    // we need to determine whether the vehicle passes the
                    // crossing from the left or the right (heuristic)
                    fromLeft = foeDistToCrossing > 0.5 * foeLane->getLength();
                } else if ((contLane && !sameSource && !ignoreIndirectBicycleTurn) || isOpposite) {
                    gap = -std::numeric_limits<double>::max(); // always break for vehicles which are on a continuation lane or for opposite-direction vehicles
                } else {
                    if (gDebugFlag1) {
                        std::cout << " distToCrossing=" << distToCrossing << " leader back=" << leaderBack << " backDist=" << leaderBackDist
                                  << " blockedStrategic=" << leader->getLaneChangeModel().isStrategicBlocked()
                                  //<< " stateRight=" << toString((LaneChangeAction)leader->getLaneChangeModel().getSavedState(-1).second)
                                  << "\n";
                    }
                    if (pastTheCrossingPoint && !sameTarget) {
                        // leader is completely past the crossing point
                        // or there is no crossing point
                        continue; // next vehicle
                    }
                    gap = distToCrossing - ego->getVehicleType().getMinGap() - leaderBackDist - foeCrossingWidth;
                    // factor 2 is to give some slack for lane-changing
                    if (gap < leader->getVehicleType().getLength() * 2 && leader->getLaneChangeModel().isStrategicBlocked()) {
                        // do not encroach on leader when it tries to change lanes
                        gap = -std::numeric_limits<double>::max();
                    }
                }
                // if the foe is already moving off the intersection, we may
                // advance up to the crossing point unless we have the same target or same source
                // (for sameSource, the crossing point indicates the point of divergence)
                const bool stopAsap = leader->isFrontOnLane(foeLane) ? cannotIgnore : (sameTarget || sameSource);
                if (gDebugFlag1) {
                    std::cout << " leader=" << leader->getID() << " contLane=" << contLane << " cannotIgnore=" << cannotIgnore << " stopAsap=" << stopAsap << "\n";
                }
                if (ignoreFoe(ego, leader)) {
                    continue;
                }
                result.emplace_back(leader, gap, stopAsap ? -1 : distToCrossing, fromLeft, inTheWay);
            }

        }
        if (ego != nullptr && MSNet::getInstance()->hasPersons()) {
            // check for crossing pedestrians (keep driving if already on top of the crossing
            const double distToPeds = distToCrossing - MSPModel::SAFETY_GAP;
            const double vehWidth = ego->getVehicleType().getWidth() + MSPModel::SAFETY_GAP; // + configurable safety gap
            /// @todo consider lateral position (depending on whether the crossing is encountered on the way in or out)
            // @check lefthand?!
            const bool wayIn = myLengthsBehindCrossing[i].first < myLaneBefore->getLength() * 0.5;
            const double vehSideOffset = (foeDistToCrossing + myLaneBefore->getWidth() * 0.5 - vehWidth * 0.5
                                          + ego->getLateralPositionOnLane() * (wayIn ? -1 : 1));
            // can access the movement model here since we already checked for existing persons above
            if (distToPeds >= -MSPModel::SAFETY_GAP && MSNet::getInstance()->getPersonControl().getMovementModel()->blockedAtDist(foeLane, vehSideOffset, vehWidth,
                    ego->getVehicleType().getParameter().getJMParam(SUMO_ATTR_JM_CROSSING_GAP, JM_CROSSING_GAP_DEFAULT),
                    collectBlockers)) {
                //if (ignoreFoe(ego, leader)) {
                //    continue;
                //}
                result.emplace_back(nullptr, -1, distToPeds);
            }
        }
    }

    //std::cout << SIMTIME << " ego=" << Named::getIDSecure(ego) << " link=" << getViaLaneOrLane()->getID() << " myWalkingAreaFoe=" << Named::getIDSecure(myWalkingAreaFoe) << "\n";
    if (ego != nullptr) {
        checkWalkingAreaFoe(ego, myWalkingAreaFoe, collectBlockers, result);
        checkWalkingAreaFoe(ego, myWalkingAreaFoeExit, collectBlockers, result);
    }

    if (MSGlobals::gLateralResolution > 0 && ego != nullptr && !isShadowLink) {
        // check for foes on the same lane
        for (std::vector<MSLane*>::const_iterator it = mySublaneFoeLanes.begin(); it != mySublaneFoeLanes.end(); ++it) {
            const MSLane* foeLane = *it;
            MSLane::AnyVehicleIterator end = foeLane->anyVehiclesEnd();
            for (MSLane::AnyVehicleIterator it_veh = foeLane->anyVehiclesBegin(); it_veh != end; ++it_veh) {
                MSVehicle* leader = (MSVehicle*)*it_veh;
                if (leader == ego) {
                    continue;
                }
                const double maxLength = MAX2(myInternalLaneBefore->getLength(), foeLane->getLength());
                const double gap = dist - maxLength - ego->getVehicleType().getMinGap() + leader->getBackPositionOnLane(foeLane);
                if (gap < -(ego->getVehicleType().getMinGap() + leader->getLength())) {
                    // ego is ahead of leader
                    continue;
                }

                const double posLat = ego->getLateralPositionOnLane();
                const double posLatLeader = leader->getLateralPositionOnLane() + leader->getLatOffset(foeLane);
                if (gDebugFlag1) {
                    std::cout << " sublaneFoe lane=" << myInternalLaneBefore->getID()
                              << " foeLane=" << foeLane->getID()
                              << " leader=" << leader->getID()
                              << " egoLane=" << ego->getLane()->getID()
                              << " leaderLane=" << leader->getLane()->getID()
                              << " egoLat=" << posLat
                              << " leaderLat=" << posLatLeader
                              << " leaderLatOffset=" << leader->getLatOffset(foeLane)
                              << " egoIndex=" << myInternalLaneBefore->getIndex()
                              << " foeIndex=" << foeLane->getIndex()
                              << " dist=" << dist
                              << " leaderBack=" << leader->getBackPositionOnLane(foeLane)
                              << "\n";
                }
                // there only is a conflict if the paths cross
                if ((posLat < posLatLeader && myInternalLaneBefore->getIndex() > foeLane->getIndex())
                        || (posLat > posLatLeader && myInternalLaneBefore->getIndex() < foeLane->getIndex())) {
                    if (gDebugFlag1) {
                        std::cout << SIMTIME << " blocked by " << leader->getID() << " (sublane split) foeLane=" << foeLane->getID() << "\n";
                    }
                    if (ignoreFoe(ego, leader)) {
                        continue;
                    }
                    result.emplace_back(leader, gap, -1);
                }
            }
        }
    }
    return result;
}


void
MSLink::checkWalkingAreaFoe(const MSVehicle* ego, const MSLane* foeLane, std::vector<const MSPerson*>* collectBlockers, LinkLeaders& result) const {
    if (foeLane != nullptr && foeLane->getEdge().getPersons().size() > 0) {
        // pedestrians may be on an arbitrary path across this
        // walkingarea. make sure to keep enough distance.
        // This is a simple but conservative solution that could be improved
        // by ignoring pedestrians that are "obviously" not on a collision course
        double distToPeds = std::numeric_limits<double>::max();
        assert(myInternalLaneBefore != nullptr);
        PositionVector egoPath = myInternalLaneBefore->getShape();
        if (ego->getLateralPositionOnLane() != 0) {
            egoPath.move2side((MSGlobals::gLefthand ? 1 : -1) * ego->getLateralPositionOnLane());
        }
        for (MSTransportable* t : foeLane->getEdge().getPersons()) {
            MSPerson* p = static_cast<MSPerson*>(t);
            const double dist = ego->getPosition().distanceTo2D(p->getPosition()) - p->getVehicleType().getLength() - MSPModel::SAFETY_GAP;
#ifdef DEBUG_WALKINGAREA
            if (ego->isSelected()) {
                std::cout << SIMTIME << " veh=" << ego->getID() << " ped=" << p->getID()
                          << " pos=" << ego->getPosition() << " pedPos=" << p->getPosition()
                          << " rawDist=" << ego->getPosition().distanceTo2D(p->getPosition())
                          << " dist=" << dist << "\n";
            }
#endif
            if (dist < ego->getVehicleType().getWidth() / 2 || isInFront(ego, egoPath, p)) {
                distToPeds = MIN2(distToPeds, dist);
                if (collectBlockers != nullptr) {
                    collectBlockers->push_back(p);
                }
            }
        }
        if (distToPeds != std::numeric_limits<double>::max()) {
            // leave extra space in front
            result.emplace_back(nullptr, -1, distToPeds - ego->getVehicleType().getMinGap());
        }
    }
}

bool
MSLink::isInFront(const MSVehicle* ego, const PositionVector& egoPath, const MSPerson* p) const {
    const double pedAngle = ego->getPosition().angleTo2D(p->getPosition());
    const double angleDiff = fabs(GeomHelper::angleDiff(ego->getAngle(), pedAngle));
#ifdef DEBUG_WALKINGAREA
    if (ego->isSelected()) {
        std::cout << " angleDiff=" << RAD2DEG(angleDiff) << "\n";
    }
#endif
    if (angleDiff < DEG2RAD(75)) {
        return egoPath.distance2D(p->getPosition()) < ego->getVehicleType().getWidth() + MSPModel::SAFETY_GAP;
    }
    return false;
}


MSLink*
MSLink::getParallelLink(int direction) const {
    if (direction == -1) {
        return myParallelRight;
    } else if (direction == 1) {
        return myParallelLeft;
    } else {
        assert(false || myLane->getOpposite() != nullptr);
        return nullptr;
    }
}

MSLink*
MSLink::getOppositeDirectionLink() const {
    if (myLane->getOpposite() != nullptr && myLaneBefore->getOpposite() != nullptr) {
        for (MSLink* cand : myLane->getOpposite()->getLinkCont()) {
            if (cand->getLane() == myLaneBefore->getOpposite()) {
                return cand;
            }
        }
    }
    return nullptr;
}


MSLink*
MSLink::computeParallelLink(int direction) {
    const MSLane* const before = getLaneBefore()->getParallelLane(direction, false);
    const MSLane* const after = getLane()->getParallelLane(direction, false);
    if (before != nullptr && after != nullptr) {
        for (MSLink* const link : before->getLinkCont()) {
            if (link->getLane() == after) {
                return link;
            }
        }
    }
    return nullptr;
}


double
MSLink::getZipperSpeed(const MSVehicle* ego, const double dist, double vSafe,
                       SUMOTime arrivalTime,
                       BlockingFoes* collectFoes) const {
    if (myFoeLinks.size() == 0) {
        // link should have LINKSTATE_MAJOR in this case
        assert(false);
        return vSafe;
    } else if (myFoeLinks.size() > 1) {
        throw ProcessError("Zipper junctions with more than two conflicting lanes are not supported (at junction '"
                           + myJunction->getID() + "')");
    }
    const double brakeGap = ego->getCarFollowModel().brakeGap(ego->getSpeed(), ego->getCarFollowModel().getMaxDecel(), 0);
    if (dist > MAX2(myFoeVisibilityDistance, brakeGap)) {
#ifdef DEBUG_ZIPPER
        const SUMOTime now = MSNet::getInstance()->getCurrentTimeStep();
        if (DEBUG_COND_ZIPPER) DEBUGOUT(SIMTIME << " getZipperSpeed ego=" << ego->getID()
                                            << " dist=" << dist << " ignoring foes (arrival in " << STEPS2TIME(arrivalTime - now) << ")\n")
#endif
            return vSafe;
    }
#ifdef DEBUG_ZIPPER
    if (DEBUG_COND_ZIPPER) DEBUGOUT(SIMTIME << " getZipperSpeed ego=" << ego->getID()
                                        << " egoAT=" << arrivalTime
                                        << " dist=" << dist
                                        << " brakeGap=" << brakeGap
                                        << " vSafe=" << vSafe
                                        << " numFoes=" << collectFoes->size()
                                        << "\n")
#endif
        MSLink* foeLink = myFoeLinks[0];
    for (const auto& item : *collectFoes) {
        const MSVehicle* foe = dynamic_cast<const MSVehicle*>(item);
        assert(foe != 0);
        const ApproachingVehicleInformation& avi = foeLink->getApproaching(foe);
        const double foeDist = (foe->isActive() ? avi.dist : MAX2(0.0, avi.dist -
                                STEPS2TIME(MSNet::getInstance()->getCurrentTimeStep() - foe->getLastActionTime()) * avi.speed));

        if (    // ignore vehicles that arrive after us (unless they are ahead and we could easily brake for them)
            ((avi.arrivalTime > arrivalTime) && !couldBrakeForLeader(dist, foeDist, ego, foe)) ||
            // also ignore vehicles that are behind us and are able to brake for us
            couldBrakeForLeader(foeDist, dist, foe, ego) ||
            // resolve ties by lane index
            (avi.arrivalTime == arrivalTime && foeDist == dist && ego->getLane()->getIndex() < foe->getLane()->getIndex())) {
#ifdef DEBUG_ZIPPER
            if (DEBUG_COND_ZIPPER) std::cout
                        << "    ignoring foe=" << foe->getID()
                        << " foeAT=" << avi.arrivalTime
                        << " foeDist=" << avi.dist
                        << " foeDist2=" << foeDist
                        << " foeSpeed=" << avi.speed
                        << " egoSpeed=" << ego->getSpeed()
                        << " deltaDist=" << foeDist - dist
                        << " delteSpeed=" << avi.speed - foe->getCarFollowModel().getMaxDecel() - ego->getSpeed()
                        << " egoCouldBrake=" << couldBrakeForLeader(dist, foeDist, ego, foe)
                        << " foeCouldBrake=" << couldBrakeForLeader(foeDist, dist, foe, ego)
                        << "\n";
#endif
            continue;
        }
        // the idea behind speed adaption is three-fold:
        // 1) ego needs to be in a car-following relationship with foe eventually
        //    thus, the ego speed should be equal to the follow speed once the foe enters
        //    the zipper junction
        // 2) ego vehicle needs to put a certain distance beteen himself and foe (safeGap)
        //    achieving this distance can be spread over time but computing
        //    safeGap is subject to estimation errors of future speeds
        // 3) deceleration can be spread out over the time until true
        //    car-following happens, at the start of speed adaptions, smaller
        //    decelerations should be sufficient

        // we cannot trust avi.arrivalSpeed if the foe has leader vehicles that are accelerating
        // lets try to extrapolate
        const double uMax = foe->getLane()->getVehicleMaxSpeed(foe);
        const double uAccel = foe->getCarFollowModel().estimateSpeedAfterDistance(foeDist, avi.speed, foe->getCarFollowModel().getMaxAccel());
        const double uEnd = MIN2(uMax, uAccel);
        const double uAvg = (avi.speed + uEnd) / 2;
        const double tf0 = foeDist / MAX2(NUMERICAL_EPS, uAvg);
        const double tf = MAX2(1.0, ceil((tf0) / TS) * TS);

        const double vMax = ego->getLane()->getVehicleMaxSpeed(ego);
        const double vAccel = ego->getCarFollowModel().estimateSpeedAfterDistance(dist, ego->getSpeed(), ego->getCarFollowModel().getMaxAccel());
        const double vDecel = ego->getCarFollowModel().estimateSpeedAfterDistance(dist, ego->getSpeed(), ego->getCarFollowModel().getMaxDecel());
        const double vEnd = MIN3(vMax, vAccel, MAX2(uEnd, vDecel));
        const double vAvg = (ego->getSpeed() + vEnd) / 2;
        const double te0 = dist / MAX2(NUMERICAL_EPS, vAvg);
        const double te = MAX2(1.0, ceil((te0) / TS) * TS);

        const double gap = dist - foe->getVehicleType().getLength() - ego->getVehicleType().getMinGap() - foeDist;
        const double safeGap = ego->getCarFollowModel().getSecureGap(ego, foe, vEnd, uEnd, foe->getCarFollowModel().getMaxDecel());
        // round t to next step size
        // increase gap to safeGap by the time foe reaches link
        // gap + u*t - (t * v + a * t^2 / 2) = safeGap
        const double deltaGap = gap + tf * uAvg - safeGap - vAvg * tf;
        const double a = 2 * deltaGap / (tf * tf);
        const double vSafeGap = ego->getSpeed() + ACCEL2SPEED(a);
        const double vFollow = ego->getCarFollowModel().followSpeed(
                                   ego, ego->getSpeed(), gap, avi.speed, foe->getCarFollowModel().getMaxDecel(), foe);

        // scale behavior based on ego time to link (te)
        const double w = MIN2(1.0, te / 10);
        const double maxDecel = w * ego->getCarFollowModel().getMaxDecel() + (1 - w) * ego->getCarFollowModel().getEmergencyDecel();
        const double vZipper = MAX3(vFollow, ego->getSpeed() - ACCEL2SPEED(maxDecel), w * vSafeGap + (1 - w) * vFollow);

        vSafe = MIN2(vSafe, vZipper);
#ifdef DEBUG_ZIPPER
        if (DEBUG_COND_ZIPPER) std::cout << "    adapting to foe=" << foe->getID()
                                             << " foeDist=" << foeDist
                                             << " foeSpeed=" << avi.speed
                                             << " foeAS=" << avi.arrivalSpeed
                                             << " egoSpeed=" << ego->getSpeed()
                                             << " uMax=" << uMax
                                             << " uAccel=" << uAccel
                                             << " uEnd=" << uEnd
                                             << " uAvg=" << uAvg
                                             << " gap=" << gap
                                             << " safeGap=" << safeGap
                                             << "\n      "
                                             << " tf=" << tf
                                             << " te=" << te
                                             << " dg=" << deltaGap
                                             << " aSafeGap=" << a
                                             << " vMax=" << vMax
                                             << " vAccel=" << vAccel
                                             << " vDecel=" << vDecel
                                             << " vEnd=" << vEnd
                                             << " vSafeGap=" << vSafeGap
                                             << " vFollow=" << vFollow
                                             << " w=" << w
                                             << " maxDecel=" << maxDecel
                                             << " vZipper=" << vZipper
                                             << " vSafe=" << vSafe
                                             << "\n";
#endif
    }
    return vSafe;
}


bool
MSLink::couldBrakeForLeader(double followDist, double leaderDist, const MSVehicle* follow, const MSVehicle* leader) {
    return (// leader is ahead of follower
               followDist > leaderDist &&
               // and follower could brake for 1 s to stay behind leader
               followDist - leaderDist > follow->getSpeed() - follow->getCarFollowModel().getMaxDecel() - leader->getSpeed());
}


void
MSLink::initParallelLinks() {
    myParallelRight = computeParallelLink(-1);
    myParallelLeft = computeParallelLink(1);
}

bool
MSLink::checkContOff() const {
    // check whether this link gets to keep its cont status switching the tls off
    // @note: this could also be pre-computed in netconvert
    // we check whether there is any major link from this edge
    for (const MSLane* cand : myLaneBefore->getEdge().getLanes()) {
        for (const MSLink* link : cand->getLinkCont()) {
            if (link->getOffState() == LINKSTATE_TL_OFF_NOSIGNAL) {
                return true;
            }
        }
    }
    return false;
}

bool
MSLink::lateralOverlap(double posLat, double width, double posLat2, double width2) {
    return fabs(posLat2 - posLat) < (width + width2) / 2;
}

std::string
MSLink::getDescription() const {
    return myLaneBefore->getID() + "->" + getViaLaneOrLane()->getID();
}


bool
MSLink::ignoreFoe(const SUMOTrafficObject* ego, const SUMOVehicle* foe) {
    if (ego == nullptr || !ego->getParameter().wasSet(VEHPARS_JUNCTIONMODEL_PARAMS_SET)) {
        return false;
    }
    const SUMOVehicleParameter& param = ego->getParameter();
    for (const std::string& typeID : StringTokenizer(param.getParameter(toString(SUMO_ATTR_JM_IGNORE_TYPES), "")).getVector()) {
        if (typeID == foe->getVehicleType().getID()) {
            return true;
        }
    }
    for (const std::string& id : StringTokenizer(param.getParameter(toString(SUMO_ATTR_JM_IGNORE_IDS), "")).getVector()) {
        if (id == foe->getID()) {
            return true;
        }
    }
    return false;
}

/****************************************************************************/
