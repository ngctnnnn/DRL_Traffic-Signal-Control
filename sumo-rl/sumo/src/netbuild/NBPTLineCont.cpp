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
/// @file    NBPTLineCont.cpp
/// @author  Gregor Laemmel
/// @author  Nikita Cherednychek
/// @date    Tue, 20 Mar 2017
///
// Container for NBPTLine during netbuild
/****************************************************************************/
#include <config.h>

#include <iostream>
#include <utils/common/MsgHandler.h>
#include <utils/common/ToString.h>
#include <utils/options/OptionsCont.h>
#include <utils/router/DijkstraRouter.h>
#include "NBPTLineCont.h"
#include "NBPTStop.h"
#include "NBEdge.h"
#include "NBNode.h"
#include "NBVehicle.h"
#include "NBPTStopCont.h"

//#define DEBUG_FIND_WAY
//#define DEBUG_CONSTRUCT_ROUTE

#define DEBUGLINEID "1986097"
#define DEBUGSTOPID ""

// ===========================================================================
// static value definitions
// ===========================================================================
const int NBPTLineCont::FWD(1);
const int NBPTLineCont::BWD(-1);
// ===========================================================================
// method definitions
// ===========================================================================

NBPTLineCont::NBPTLineCont() { }


NBPTLineCont::~NBPTLineCont() {
    for (auto& myPTLine : myPTLines) {
        delete myPTLine.second;
    }
    myPTLines.clear();
}

void
NBPTLineCont::insert(NBPTLine* ptLine) {
    myPTLines[ptLine->getLineID()] = ptLine;
}

void NBPTLineCont::process(NBEdgeCont& ec, NBPTStopCont& sc, bool routeOnly) {
    const bool silent = routeOnly;
    for (auto& item : myPTLines) {
        NBPTLine* line = item.second;
        if (item.second->getMyWays().size() > 0) {
            // loaded from OSM rather than ptline input. We can use extra
            // information to reconstruct route and stops
            constructRoute(line, ec, silent);
            if (!routeOnly) {
                // map stops to ways, using the constructed route for loose stops
                reviseStops(line, ec, sc);
            }
        }
        line->deleteInvalidStops(ec, sc);
        //line->deleteDuplicateStops();
        for (NBPTStop* stop : line->getStops()) {
            myServedPTStops.insert(stop->getID());
        }
    }
}

void
NBPTLineCont::reviseStops(NBPTLine* line, const NBEdgeCont& ec, NBPTStopCont& sc) {
    const std::vector<std::string>& waysIds = line->getMyWays();
    if (waysIds.size() == 1 && line->getStops().size() > 1) {
        reviseSingleWayStops(line, ec, sc);
        return;
    }
    if (waysIds.size() <= 1) {
        WRITE_WARNINGF("Cannot revise pt stop localization for pt line '%', which consist of one way only. Ignoring!", line->getLineID());
        return;
    }
    if (line->getRoute().size() == 0) {
        WRITE_WARNINGF("Cannot revise pt stop localization for pt line '%', which has no route edges. Ignoring!", line->getLineID());
        return;
    }
    std::vector<NBPTStop*> stops = line->getStops();
    for (NBPTStop* stop : stops) {
        //get the corresponding and one of the two adjacent ways
        stop = findWay(line, stop, ec, sc);
        if (stop == nullptr) {
            // warning already given
            continue;
        }
        auto waysIdsIt = std::find(waysIds.begin(), waysIds.end(), stop->getOrigEdgeId());
        if (waysIdsIt == waysIds.end()) {
            // warning already given
            continue;
        }
        // find directional edge (OSM ways are bidirectional)
        std::vector<long long int>* way = line->getWaysNodes(stop->getOrigEdgeId());
        if (way == nullptr) {
            WRITE_WARNINGF("Cannot assign stop '%' on edge '%' to pt line '%' (wayNodes not found). Ignoring!",
                           stop->getID(), stop->getOrigEdgeId(), line->getLineID());
            continue;
        }


        int dir;
        std::string adjIdPrev;
        std::string adjIdNext;
        if (waysIdsIt != waysIds.begin()) {
            adjIdPrev = *(waysIdsIt - 1);
        }
        if (waysIdsIt != (waysIds.end() - 1)) {
            adjIdNext = *(waysIdsIt + 1);
        }
        std::vector<long long int>* wayPrev = line->getWaysNodes(adjIdPrev);
        std::vector<long long int>* wayNext = line->getWaysNodes(adjIdNext);
        if (wayPrev == nullptr && wayNext == nullptr) {
            WRITE_WARNINGF("Cannot revise pt stop localization for incomplete pt line '%'. Ignoring!", line->getLineID());
            continue;
        }
        long long int wayEnds = *(way->end() - 1);
        long long int wayBegins = *(way->begin());
        long long int wayPrevEnds = wayPrev != nullptr ? *(wayPrev->end() - 1) : 0;
        long long int wayPrevBegins = wayPrev != nullptr ? *(wayPrev->begin()) : 0;
        long long int wayNextEnds = wayNext != nullptr ? *(wayNext->end() - 1) : 0;
        long long int wayNextBegins = wayNext != nullptr ? *(wayNext->begin()) : 0;
        if (wayBegins == wayPrevEnds || wayBegins == wayPrevBegins || wayEnds == wayNextBegins
                || wayEnds == wayNextEnds) {
            dir = FWD;
        } else if (wayEnds == wayPrevBegins || wayEnds == wayPrevEnds || wayBegins == wayNextEnds
                   || wayBegins == wayNextBegins) {
            dir = BWD;
        } else {
            WRITE_WARNINGF("Cannot revise pt stop localization for incomplete pt line '%'. Ignoring!", line->getLineID());
            continue;
        }

        std::string edgeId = stop->getEdgeId();
        NBEdge* current = ec.getByID(edgeId);
        int assignedTo = edgeId.at(0) == '-' ? BWD : FWD;

        if (dir != assignedTo) {
            NBEdge* reverse = NBPTStopCont::getReverseEdge(current);
            if (reverse == nullptr) {
                WRITE_WARNINGF("Could not re-assign PT stop '%', probably broken osm file.", stop->getID());
                continue;
            }
            if (stop->getLines().size() > 0) {
                NBPTStop* reverseStop = sc.getReverseStop(stop, ec);
                sc.insert(reverseStop);
                line->replaceStop(stop, reverseStop);
                stop = reverseStop;
            } else {
                WRITE_WARNINGF("PT stop '%' has been moved to edge '%'.", stop->getID(), reverse->getID());
            }
            stop->setEdgeId(reverse->getID(), ec);
        }
        stop->addLine(line->getRef());
    }
}


void NBPTLineCont::reviseSingleWayStops(NBPTLine* line, const NBEdgeCont& ec, NBPTStopCont& sc) {
    const std::vector<std::string>& waysIds = line->getMyWays();
    for (NBPTStop* stop : line->getStops()) {
        //get the corresponding and one of the two adjacent ways
        stop = findWay(line, stop, ec, sc);
        if (stop == nullptr) {
            // warning already given
            continue;
        }
        auto waysIdsIt = std::find(waysIds.begin(), waysIds.end(), stop->getOrigEdgeId());
        if (waysIdsIt == waysIds.end()) {
            // warning already given
            continue;
        }
        stop->addLine(line->getRef());
    }

}

NBPTStop*
NBPTLineCont::findWay(NBPTLine* line, NBPTStop* stop, const NBEdgeCont& ec, NBPTStopCont& sc) const {
    const std::vector<std::string>& waysIds = line->getMyWays();
#ifdef DEBUG_FIND_WAY
    if (stop->getID() == DEBUGSTOPID) {
        std::cout << " stop=" << stop->getID() << " line=" << line->getLineID() << " edgeID=" << stop->getEdgeId() << " origID=" << stop->getOrigEdgeId() << "\n";
    }
#endif
    if (stop->isLoose()) {
        // find closest edge in route
        double minDist = std::numeric_limits<double>::max();
        NBEdge* best = nullptr;
        for (NBEdge* edge : line->getRoute()) {
            const double dist = edge->getLaneShape(0).distance2D(stop->getPosition());
            if (dist < minDist) {
                best = edge;
                minDist = dist;
            }
        }
#ifdef DEBUG_FIND_WAY
        if (stop->getID() == DEBUGSTOPID) {
            std::cout << "   best=" << Named::getIDSecure(best) << " minDist=" << minDist << " wayID=" << getWayID(best->getID())
                      << " found=" << (std::find(waysIds.begin(), waysIds.end(), getWayID(best->getID())) != waysIds.end())
                      << " wayIDs=" << toString(waysIds) << "\n";
        }
#endif
        if (minDist < OptionsCont::getOptions().getFloat("ptline.match-dist")) {
            const std::string wayID = getWayID(best->getID());
            if (stop->getEdgeId() == "") {
                stop->setEdgeId(best->getID(), ec);
                stop->setOrigEdgeId(wayID);
            } else if (stop->getEdgeId() != best->getID()) {
                // stop is used by multiple lines and mapped to different edges.
                // check if an alterantive stop already exists
                NBPTStop* newStop = sc.findStop(wayID, stop->getPosition());
                if (newStop == nullptr) {
                    newStop = new NBPTStop(stop->getID() + "@" + line->getLineID(), stop->getPosition(), best->getID(), wayID, stop->getLength(), stop->getName(), stop->getPermissions());
                    newStop->setEdgeId(best->getID(), ec);  // trigger lane assignment
                    sc.insert(newStop);
                }
                line->replaceStop(stop, newStop);
                stop = newStop;
            }
        } else {
            WRITE_WARNINGF("Could not assign stop '%' to pt line '%' (closest edge '%', distance %). Ignoring!",
                           stop->getID(), line->getLineID(), Named::getIDSecure(best), minDist);
            return nullptr;
        }
    } else {
        // if the stop is part of an edge, find that edge among the line edges
        auto waysIdsIt = waysIds.begin();
        for (; waysIdsIt != waysIds.end(); waysIdsIt++) {
            if ((*waysIdsIt) == stop->getOrigEdgeId()) {
                break;
            }
        }

        if (waysIdsIt == waysIds.end()) {
            // stop edge not found, try additional edges
            for (auto& edgeCand : stop->getAdditionalEdgeCandidates()) {
                bool found = false;
                waysIdsIt =  waysIds.begin();
                for (; waysIdsIt != waysIds.end(); waysIdsIt++) {
                    if ((*waysIdsIt) == edgeCand.first) {
                        if (stop->setEdgeId(edgeCand.second, ec)) {
                            stop->setOrigEdgeId(edgeCand.first);
                            found = true;
                            break;
                        }
                    }
                }
                if (found) {
                    break;
                }
            }
            if (waysIdsIt == waysIds.end()) {
                WRITE_WARNINGF("Cannot assign stop % on edge '%' to pt line '%'. Ignoring!", stop->getID(), stop->getOrigEdgeId(), line->getLineID());
            }
        }
    }
    return stop;
}


void NBPTLineCont::constructRoute(NBPTLine* pTLine, const NBEdgeCont& cont, bool silent) {
    std::vector<NBEdge*> edges;

    NBNode* first = nullptr;
    NBNode* last = nullptr;
    std::vector<NBEdge*> prevWayEdges;
    std::vector<NBEdge*> prevWayMinusEdges;
    prevWayEdges.clear();
    prevWayMinusEdges.clear();
    std::vector<NBEdge*> currentWayEdges;
    std::vector<NBEdge*> currentWayMinusEdges;
    for (auto it3 = pTLine->getMyWays().begin();
            it3 != pTLine->getMyWays().end(); it3++) {

        if (cont.retrieve(*it3, false) != nullptr) {
            currentWayEdges.push_back(cont.retrieve(*it3, false));
        } else {
            int i = 0;
            while (cont.retrieve(*it3 + "#" + std::to_string(i), true) != nullptr) {
                if (cont.retrieve(*it3 + "#" + std::to_string(i), false)) {
                    currentWayEdges.push_back(cont.retrieve(*it3 + "#" + std::to_string(i), false));
                }
                i++;
            }
        }

        if (cont.retrieve("-" + *it3, false) != nullptr) {
            currentWayMinusEdges.push_back(cont.retrieve("-" + *it3, false));
        } else {
            int i = 0;
            while (cont.retrieve("-" + *it3 + "#" + std::to_string(i), true) != nullptr) {
                if (cont.retrieve("-" + *it3 + "#" + std::to_string(i), false)) {
                    currentWayMinusEdges.insert(currentWayMinusEdges.begin(),
                                                cont.retrieve("-" + *it3 + "#" + std::to_string(i), false));
                }
                i++;
            }
        }
#ifdef DEBUG_CONSTRUCT_ROUTE
        if (pTLine->getLineID() == DEBUGLINEID) {
            std::cout << " way=" << (*it3)
                      << " done=" << toString(edges)
                      << " first=" << Named::getIDSecure(first)
                      << " last=" << Named::getIDSecure(last)
                      << " +=" << toString(currentWayEdges)
                      << " -=" << toString(currentWayMinusEdges)
                      << "\n";
        }
#endif
        if (currentWayEdges.empty()) {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if0\n";
            }
#endif
            continue;
        }
        if (last == currentWayEdges.front()->getFromNode() && last != nullptr) {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if1\n";
            }
#endif
            if (!prevWayEdges.empty()) {
                edges.insert(edges.end(), prevWayEdges.begin(), prevWayEdges.end());
                prevWayEdges.clear();
                prevWayMinusEdges.clear();
            }
            edges.insert(edges.end(), currentWayEdges.begin(), currentWayEdges.end());
            last = currentWayEdges.back()->getToNode();
        } else if (last == currentWayEdges.back()->getToNode() && last != nullptr) {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if2\n";
            }
#endif
            if (!prevWayEdges.empty()) {
                edges.insert(edges.end(), prevWayEdges.begin(), prevWayEdges.end());
                prevWayEdges.clear();
                prevWayMinusEdges.clear();
            }
            if (currentWayMinusEdges.empty()) {
                currentWayEdges.clear();
                last = nullptr;
#ifdef DEBUG_CONSTRUCT_ROUTE
                if (pTLine->getLineID() == DEBUGLINEID) {
                    std::cout << " continue1\n";
                }
#endif
                continue;
            } else {
                edges.insert(edges.end(), currentWayMinusEdges.begin(), currentWayMinusEdges.end());
                last = currentWayMinusEdges.back()->getToNode();
            }
        } else if (first == currentWayEdges.front()->getFromNode() && first != nullptr) {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if3\n";
            }
#endif
            edges.insert(edges.end(), prevWayMinusEdges.begin(), prevWayMinusEdges.end());
            edges.insert(edges.end(), currentWayEdges.begin(), currentWayEdges.end());
            last = currentWayEdges.back()->getToNode();
            prevWayEdges.clear();
            prevWayMinusEdges.clear();
        } else if (first == currentWayEdges.back()->getToNode() && first != nullptr) {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if4\n";
            }
#endif
            edges.insert(edges.end(), prevWayMinusEdges.begin(), prevWayMinusEdges.end());
            if (currentWayMinusEdges.empty()) {
                currentWayEdges.clear();
                last = nullptr;
                prevWayEdges.clear();
                prevWayMinusEdges.clear();
#ifdef DEBUG_CONSTRUCT_ROUTE
                if (pTLine->getLineID() == DEBUGLINEID) {
                    std::cout << " continue2\n";
                }
#endif
                continue;
            } else {
                edges.insert(edges.end(), currentWayMinusEdges.begin(), currentWayMinusEdges.end());
                last = currentWayMinusEdges.back()->getToNode();
                prevWayEdges.clear();
                prevWayMinusEdges.clear();
            }
        } else {
#ifdef DEBUG_CONSTRUCT_ROUTE
            if (pTLine->getLineID() == DEBUGLINEID) {
                std::cout << " if5\n";
            }
#endif
            if (it3 != pTLine->getMyWays().begin()) {
                if (!silent) {
                    WRITE_WARNINGF("Incomplete route for pt line '%'%.", pTLine->getLineID(),
                                   (pTLine->getName() != "" ? " (" + pTLine->getName() + ")" : ""));
                }
            } else if (pTLine->getMyWays().size() == 1) {
                if (currentWayEdges.size() > 0) {
                    edges.insert(edges.end(), currentWayEdges.begin(), currentWayEdges.end());
                } else {
                    edges.insert(edges.end(), currentWayMinusEdges.begin(), currentWayMinusEdges.end());
                }
            }
            prevWayEdges = currentWayEdges;
            prevWayMinusEdges = currentWayMinusEdges;
            if (!prevWayEdges.empty()) {
                first = prevWayEdges.front()->getFromNode();
                last = prevWayEdges.back()->getToNode();
            } else {
                first = nullptr;
                last = nullptr;
            }
        }
        currentWayEdges.clear();
        currentWayMinusEdges.clear();
    }
    pTLine->setEdges(edges);
}


void
NBPTLineCont::addEdges2Keep(const OptionsCont& oc, std::set<std::string>& into) {
    if (oc.isSet("ptline-output")) {
        for (auto& item : myPTLines) {
            for (auto edge : item.second->getRoute()) {
                into.insert(edge->getID());
            }
        }
    }
}


void
NBPTLineCont::replaceEdge(const std::string& edgeID, const EdgeVector& replacement) {
    //std::cout << " replaceEdge " << edgeID << " replacement=" << toString(replacement) << "\n";
    if (myPTLines.size() > 0 && myPTLineLookup.size() == 0) {
        // init lookup once
        for (auto& item : myPTLines) {
            for (const NBEdge* e : item.second->getRoute()) {
                myPTLineLookup[e->getID()].insert(item.second);
            }
        }
    }
    for (NBPTLine* line : myPTLineLookup[edgeID]) {
        line->replaceEdge(edgeID, replacement);
        for (const NBEdge* e : replacement) {
            myPTLineLookup[e->getID()].insert(line);
        }
    }
    myPTLineLookup.erase(edgeID);
}


std::set<std::string>&
NBPTLineCont::getServedPTStops() {
    return myServedPTStops;
}


void
NBPTLineCont::fixBidiStops(const NBEdgeCont& ec) {
    std::map<std::string, SUMOVehicleClass> types;
    types["bus"] = SVC_BUS;
    types["tram"] = SVC_TRAM;
    types["train"] = SVC_RAIL;
    types["subway"] = SVC_RAIL_URBAN;
    types["light_rail"] = SVC_RAIL_URBAN;
    types["ferry"] = SVC_SHIP;

    SUMOAbstractRouter<NBRouterEdge, NBVehicle>* const router = new DijkstraRouter<NBRouterEdge, NBVehicle>(
        ec.getAllRouterEdges(), true, &NBRouterEdge::getTravelTimeStatic, nullptr, true);

    for (auto& item : myPTLines) {
        NBPTLine* line = item.second;
        std::vector<NBPTStop*> stops = line->getStops();
        if (stops.size() < 2) {
            continue;
        }
        if (types.count(line->getType()) == 0) {
            WRITE_WARNINGF("Could not determine vehicle class for public transport line of type '%'.", line->getType());
            continue;
        }
        NBVehicle veh(line->getRef(), types[line->getType()]);
        std::vector<NBPTStop*> newStops;
        NBPTStop* from = nullptr;
        for (auto it = stops.begin(); it != stops.end(); ++it) {
            NBPTStop* to = *it;
            NBPTStop* used = *it;
            if (to->getBidiStop() != nullptr) {
                double best = std::numeric_limits<double>::max();
                NBPTStop* to2 = to->getBidiStop();
                if (from == nullptr) {
                    if ((it + 1) != stops.end()) {
                        from = to;
                        NBPTStop* from2 = to2;
                        to = *(it + 1);
                        const double c1 = getCost(ec, *router, from, to, &veh);
                        const double c2 = getCost(ec, *router, from2, to, &veh);
                        //std::cout << " from=" << from->getID() << " to=" << to->getID() << " c1=" << MIN2(10000.0, c1) << "\n";
                        //std::cout << " from2=" << from2->getID() << " to=" << to->getID() << " c2=" << MIN2(10000.0, c2) << "\n";
                        best = c1;
                        if (to->getBidiStop() != nullptr) {
                            to2 = to->getBidiStop();
                            const double c3 = getCost(ec, *router, from, to2, &veh);
                            const double c4 = getCost(ec, *router, from2, to2, &veh);
                            //std::cout << " from=" << from->getID() << " to2=" << to2->getID() << " c3=" << MIN2(10000.0, c3) << "\n";
                            //std::cout << " from2=" << from2->getID() << " to2=" << to2->getID() << " c4=" << MIN2(10000.0, c4) << "\n";
                            if (c2 < best) {
                                used = from2;
                                best = c2;
                            }
                            if (c3 < best) {
                                used = from;
                                best = c3;
                            }
                            if (c4 < best) {
                                used = from2;
                                best = c4;
                            }
                        } else {
                            if (c2 < c1) {
                                used = from2;
                                best = c2;
                            } else {
                                best = c1;
                            }
                        }
                    }
                } else {
                    const double c1 = getCost(ec, *router, from, to, &veh);
                    const double c2 = getCost(ec, *router, from, to2, &veh);
                    //std::cout << " from=" << from->getID() << " to=" << to->getID() << " c1=" << MIN2(10000.0, c1) << "\n";
                    //std::cout << " from=" << from->getID() << " t2o=" << to2->getID() << " c2=" << MIN2(10000.0, c2) << "\n";
                    if (c2 < c1) {
                        used = to2;
                        best = c2;
                    } else {
                        best = c1;
                    }

                }
                if (best < std::numeric_limits<double>::max()) {
                    from = used;
                } else {
                    WRITE_WARNINGF("Could not determine direction for line '%' at stop '%'.", line->getLineID(), used->getID());
                }
            }
            from = used;
            newStops.push_back(used);
        }
        assert(stops.size() == newStops.size());
        line->replaceStops(newStops);
    }
    delete router;
}


void
NBPTLineCont::removeInvalidEdges(const NBEdgeCont& ec) {
    for (auto& item : myPTLines) {
        item.second->removeInvalidEdges(ec);
    }
}


void
NBPTLineCont::fixPermissions() {
    for (auto& item : myPTLines) {
        NBPTLine* line = item.second;
        const std::vector<NBEdge*>& route = line->getRoute();
        const SUMOVehicleClass svc = line->getVClass();
        for (int i = 1; i < (int)route.size(); i++) {
            NBEdge* e1 = route[i - 1];
            NBEdge* e2 = route[i];
            std::vector<NBEdge::Connection> cons = e1->getConnectionsFromLane(-1, e2, -1);
            if (cons.size() == 0) {
                //WRITE_WARNINGF("Disconnected ptline '%' between edge '%' and edge '%'", line->getLineID(), e1->getID(), e2->getID());
            } else {
                bool ok = false;
                for (const auto& c : cons) {
                    if ((e1->getPermissions(c.fromLane) & svc) == svc) {
                        ok = true;
                        break;
                    }
                }
                if (!ok) {
                    int lane = cons[0].fromLane;
                    e1->setPermissions(e1->getPermissions(lane) | svc, lane);
                }
            }
        }
    }
}

double
NBPTLineCont::getCost(const NBEdgeCont& ec, SUMOAbstractRouter<NBRouterEdge, NBVehicle>& router,
                      const NBPTStop* from, const NBPTStop* to, const NBVehicle* veh) {
    NBEdge* fromEdge = ec.getByID(from->getEdgeId());
    NBEdge* toEdge = ec.getByID(to->getEdgeId());
    if (fromEdge == nullptr || toEdge == nullptr) {
        return std::numeric_limits<double>::max();
    } else if (fromEdge == toEdge) {
        if (from->getEndPos() <= to->getEndPos()) {
            return to->getEndPos() - from->getEndPos();
        } else {
            return std::numeric_limits<double>::max();
        }
    }
    std::vector<const NBRouterEdge*> route;
    router.compute(fromEdge, toEdge, veh, 0, route);
    if (route.size() == 0) {
        return std::numeric_limits<double>::max();
    } else {
        return router.recomputeCosts(route, veh, 0);
    }
}


std::string
NBPTLineCont::getWayID(const std::string& edgeID) {
    std::size_t found = edgeID.rfind("#");
    std::string result = edgeID;
    if (found != std::string::npos) {
        result = edgeID.substr(0, found);
    }
    if (result[0] == '-') {
        result = result.substr(1);
    }
    return result;
}
