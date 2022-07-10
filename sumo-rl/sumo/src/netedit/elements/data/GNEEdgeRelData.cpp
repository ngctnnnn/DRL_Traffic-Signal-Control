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
/// @file    GNEEdgeRelData.cpp
/// @author  Pablo Alvarez Lopez
/// @date    Jan 2020
///
// class for edge relation data
/****************************************************************************/


// ===========================================================================
// included modules
// ===========================================================================
#include <config.h>

#include <netedit/GNENet.h>
#include <netedit/GNEUndoList.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNEViewParent.h>
#include <netedit/changes/GNEChange_Attribute.h>
#include <netedit/frames/data/GNEEdgeRelDataFrame.h>
#include <utils/gui/div/GLHelper.h>
#include <utils/gui/globjects/GLIncludes.h>

#include "GNEEdgeRelData.h"
#include "GNEDataInterval.h"


// ===========================================================================
// member method definitions
// ===========================================================================

// ---------------------------------------------------------------------------
// GNEEdgeRelData - methods
// ---------------------------------------------------------------------------

GNEEdgeRelData::GNEEdgeRelData(GNEDataInterval* dataIntervalParent, GNEEdge* fromEdge, GNEEdge* toEdge,
                               const Parameterised::Map& parameters) :
    GNEGenericData(SUMO_TAG_EDGEREL, GLO_EDGERELDATA, dataIntervalParent, parameters,
{}, {fromEdge, toEdge}, {}, {}, {}, {}) {
}


GNEEdgeRelData::~GNEEdgeRelData() {}


const RGBColor&
GNEEdgeRelData::getColor() const {
    if (myNet->getViewNet()->getEditModes().dataEditMode == DataEditMode::DATA_EDGERELDATA) {
        // get selected data interval and filtered attribute
        const GNEDataInterval* dataInterval = myNet->getViewNet()->getViewParent()->getEdgeRelDataFrame()->getIntervalSelector()->getDataInterval();
        const std::string filteredAttribute = myNet->getViewNet()->getViewParent()->getEdgeRelDataFrame()->getAttributeSelector()->getFilteredAttribute();
        // continue if there is a selected data interval and filtered attribute
        if (dataInterval && (filteredAttribute.size() > 0)) {
            // obtain minimum and maximum value
            const double minValue = dataInterval->getSpecificAttributeColors().at(myTagProperty.getTag()).getMinValue(filteredAttribute);
            const double maxValue = dataInterval->getSpecificAttributeColors().at(myTagProperty.getTag()).getMaxValue(filteredAttribute);
            // get value
            const double value = parse<double>(getParameter(filteredAttribute, "0"));
            // return color
            return GNEViewNetHelper::getRainbowScaledColor(minValue, maxValue, value);
        }
    }
    return RGBColor::GREEN;
}


bool
GNEEdgeRelData::isGenericDataVisible() const {
    // obtain pointer to edge data frame (only for code legibly)
    const GNEEdgeRelDataFrame* edgeRelDataFrame = myNet->getViewNet()->getViewParent()->getEdgeRelDataFrame();
    // get current data edit mode
    DataEditMode dataMode = myNet->getViewNet()->getEditModes().dataEditMode;
    // check if we have to filter generic data
    if ((dataMode == DataEditMode::DATA_INSPECT) || (dataMode == DataEditMode::DATA_DELETE) || (dataMode == DataEditMode::DATA_SELECT)) {
        return isVisibleInspectDeleteSelect();
    } else if (edgeRelDataFrame->shown()) {
        // check interval
        if ((edgeRelDataFrame->getIntervalSelector()->getDataInterval() != nullptr) &&
                (edgeRelDataFrame->getIntervalSelector()->getDataInterval() != myDataIntervalParent)) {
            return false;
        }
        // check attribute
        if ((edgeRelDataFrame->getAttributeSelector()->getFilteredAttribute().size() > 0) &&
                (getParametersMap().count(edgeRelDataFrame->getAttributeSelector()->getFilteredAttribute()) == 0)) {
            return false;
        }
        // all checks ok, then return true
        return true;
    } else {
        // GNEEdgeRelDataFrame hidden, then return false
        return false;
    }
}


void
GNEEdgeRelData::updateGeometry() {
    // just compute path
    computePathElement();
}


void
GNEEdgeRelData::drawGL(const GUIVisualizationSettings& /*s*/) const {
    // Nothing to draw
}


double
GNEEdgeRelData::getExaggeration(const GUIVisualizationSettings& /*s*/) const {
    return 1;
}


void
GNEEdgeRelData::computePathElement() {
    // calculate path
    myNet->getPathManager()->calculateConsecutivePathEdges(this, SVC_IGNORING, getParentEdges());
}


void
GNEEdgeRelData::drawPartialGL(const GUIVisualizationSettings& s, const GNELane* lane, const GNEPathManager::Segment* /*segment*/, const double offsetFront) const {
    if (myNet->getViewNet()->getEditModes().isCurrentSupermodeData()) {
        // get flag for only draw contour
        const bool onlyDrawContour = !isGenericDataVisible();
        // get lane width
        const double laneWidth = s.addSize.getExaggeration(s, lane) * (lane->getParentEdge()->getNBEdge()->getLaneWidth(lane->getIndex()) * 0.5);
        // Start drawing adding an gl identificator
        if (!onlyDrawContour) {
            GLHelper::pushName(getGlID());
        }
        // Add a draw matrix
        GLHelper::pushMatrix();
        // Start with the drawing of the area translating matrix to origin
        myNet->getViewNet()->drawTranslateFrontAttributeCarrier(this, GLO_EDGERELDATA, offsetFront);
        // Set orange color
        GLHelper::setColor(RGBColor::BLACK);
        // draw box lines
        GUIGeometry::drawLaneGeometry(s, myNet->getViewNet()->getPositionInformation(), lane->getLaneShape(), lane->getShapeRotations(), lane->getShapeLengths(), {}, laneWidth, onlyDrawContour);
        // translate to top
        glTranslated(0, 0, 0.01);
        // Set color
        if (isAttributeCarrierSelected()) {
            GLHelper::setColor(s.colorSettings.selectedEdgeDataColor);
        } else {
            GLHelper::setColor(getColor());
        }
        // draw interne box lines
        GUIGeometry::drawLaneGeometry(s, myNet->getViewNet()->getPositionInformation(), lane->getLaneShape(), lane->getShapeRotations(), lane->getShapeLengths(), {}, laneWidth - 0.1, onlyDrawContour);
        // Pop last matrix
        GLHelper::popMatrix();
        // Pop name
        if (!onlyDrawContour) {
            GLHelper::popName();
        }
        // draw lock icon
        GNEViewNetHelper::LockIcon::drawLockIcon(this, getType(), getPositionInView(), 1);
        // draw filtered attribute
        if (getParentEdges().front()->getLanes().front() == lane) {
            drawFilteredAttribute(s, lane->getLaneShape(), 
                myNet->getViewNet()->getViewParent()->getEdgeRelDataFrame()->getAttributeSelector()->getFilteredAttribute(),
                myNet->getViewNet()->getViewParent()->getEdgeRelDataFrame()->getIntervalSelector()->getDataInterval());
        }
        // draw dotted contour
        if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
            if (getParentEdges().front() == lane->getParentEdge()) {
                GNEEdge::drawDottedContourEdge(GUIDottedGeometry::DottedContourType::INSPECT, s, getParentEdges().front(), true, false);
            } else {
                GNEEdge::drawDottedContourEdge(GUIDottedGeometry::DottedContourType::INSPECT, s, getParentEdges().back(), false, true);
            }
        }
    }
}


void
GNEEdgeRelData::drawPartialGL(const GUIVisualizationSettings& s, const GNELane* fromLane, const GNELane* toLane, const GNEPathManager::Segment* /*segment*/, const double offsetFront) const {
    if (myNet->getViewNet()->getEditModes().isCurrentSupermodeData()) {
        // get flag for only draw contour
        const bool onlyDrawContour = !isGenericDataVisible();
        if ((fromLane->getParentEdge() == getParentEdges().front()) && (toLane->getParentEdge() == getParentEdges().back()) &&
                (getParentEdges().front() != getParentEdges().back())) {
            // Start drawing adding an gl identificator
            if (!onlyDrawContour) {
                GLHelper::pushName(getGlID());
            }
            // draw lanes
            const auto fromLanes = fromLane->getParentEdge()->getLanes();
            const auto toLanes = toLane->getParentEdge()->getLanes();
            size_t index = 0;
            while ((index < fromLanes.size()) || (index < toLanes.size())) {
                // get lanes
                const GNELane* from = (index < fromLanes.size()) ? fromLanes.at(index) : fromLanes.back();
                const GNELane* to = (index < toLanes.size()) ? toLanes.at(index) : toLanes.back();
                // get lane widths
                const double laneWidthFrom = s.addSize.getExaggeration(s, from) * (from->getParentEdge()->getNBEdge()->getLaneWidth(from->getIndex()) * 0.5);
                const double laneWidthTo = s.addSize.getExaggeration(s, to) * (to->getParentEdge()->getNBEdge()->getLaneWidth(to->getIndex()) * 0.5);
                const double laneWidth = laneWidthFrom < laneWidthTo ? laneWidthFrom : laneWidthTo;
                // Add a draw matrix
                GLHelper::pushMatrix();
                // translate to GLO
                glTranslated(0, 0, getType() + offsetFront);
                // Set color
                GLHelper::setColor(RGBColor::BLACK);
                if (from->getLane2laneConnections().exist(to)) {
                    // draw box lines
                    GUIGeometry::drawContourGeometry(from->getLane2laneConnections().getLane2laneGeometry(to), laneWidth);
                    // translate to top
                    glTranslated(0, 0, 0.01);
                    // Set color
                    if (isAttributeCarrierSelected()) {
                        GLHelper::setColor(s.colorSettings.selectedEdgeDataColor);
                    } else {
                        GLHelper::setColor(getColor());
                    }
                    // draw interne box lines
                    GUIGeometry::drawContourGeometry(from->getLane2laneConnections().getLane2laneGeometry(to), laneWidth - 0.1);
                } else {
                    // draw line between end of first shape and first position of second shape
                    GLHelper::drawBoxLines({from->getLaneShape().back(), to->getLaneShape().front()}, laneWidth);
                    // translate to top
                    glTranslated(0, 0, 0.01);
                    // Set color
                    if (isAttributeCarrierSelected()) {
                        GLHelper::setColor(s.colorSettings.selectedEdgeDataColor);
                    } else {
                        GLHelper::setColor(getColor());
                    }
                    // draw interne line between end of first shape and first position of second shape
                    GLHelper::drawBoxLines({from->getLaneShape().back(), to->getLaneShape().front()}, laneWidth - 0.1);
                }
                // Pop last matrix
                GLHelper::popMatrix();
                // update index
                index++;
            }
            // Pop name
            if (!onlyDrawContour) {
                GLHelper::popName();
            }
            // draw dotted contour
            if (myNet->getViewNet()->isAttributeCarrierInspected(this)) {
                // declare lanes
                const GNELane* laneTopA = getParentEdges().front()->getLanes().front();
                const GNELane* laneTopB = getParentEdges().back()->getLanes().front();
                const GNELane* laneBotA = getParentEdges().front()->getLanes().back();
                const GNELane* laneBotB = getParentEdges().back()->getLanes().back();
                // declare LaneDrawingConstants
                GNELane::LaneDrawingConstants laneDrawingConstantsTop(s, laneTopA);
                GNELane::LaneDrawingConstants laneDrawingConstantsBot(s, laneBotA);
                // declare DottedGeometryColor
                GUIDottedGeometry::DottedGeometryColor dottedGeometryColor(s);
                // Push draw matrix
                GLHelper::pushMatrix();
                // translate to front
                glTranslated(0, 0, GLO_DOTTEDCONTOUR_INSPECTED);
                // check if lane2lane connection exist
                if (laneTopA->getLane2laneConnections().exist(laneTopB)) {
                    // obtain lane2lane top dotted geometry
                    GUIDottedGeometry lane2lane(s, laneTopA->getLane2laneConnections().getLane2laneGeometry(laneTopB).getShape(), false);
                    // move shape to side
                    lane2lane.moveShapeToSide(laneDrawingConstantsTop.halfWidth);
                    // invert offset
                    lane2lane.invertOffset();
                    // reset dottedGeometryColor
                    dottedGeometryColor.reset();
                    // draw top dotted geometry
                    lane2lane.drawDottedGeometry(dottedGeometryColor, GUIDottedGeometry::DottedContourType::INSPECT);
                } else {
                    // create dotted geometry using lane extremes
                    GUIDottedGeometry dottedGeometry(s, {laneTopA->getLaneShape().back(), laneTopB->getLaneShape().front()}, false);
                    // move shape to side
                    dottedGeometry.moveShapeToSide(laneDrawingConstantsTop.halfWidth);
                    // invert offset
                    dottedGeometry.invertOffset();
                    // reset dottedGeometryColor
                    dottedGeometryColor.reset();
                    // draw top dotted geometry
                    dottedGeometry.drawDottedGeometry(dottedGeometryColor, GUIDottedGeometry::DottedContourType::INSPECT);
                }
                // check if lane2lane bot connection exist
                if (laneBotA->getLane2laneConnections().exist(laneBotB)) {
                    // obtain lane2lane dotted geometry
                    GUIDottedGeometry lane2lane(s, laneBotA->getLane2laneConnections().getLane2laneGeometry(laneBotB).getShape(), false);
                    // move shape to side
                    lane2lane.moveShapeToSide(laneDrawingConstantsBot.halfWidth * -1);
                    // reset dottedGeometryColor
                    dottedGeometryColor.reset();
                    // draw top dotted geometry
                    lane2lane.drawDottedGeometry(dottedGeometryColor, GUIDottedGeometry::DottedContourType::INSPECT);
                } else {
                    // create dotted geometry using lane extremes
                    GUIDottedGeometry dottedGeometry(s, {laneBotA->getLaneShape().back(), laneBotB->getLaneShape().front()}, false);
                    // move shape to side
                    dottedGeometry.moveShapeToSide(laneDrawingConstantsBot.halfWidth * -1);
                    // reset dottedGeometryColor
                    dottedGeometryColor.reset();
                    // draw top dotted geometry
                    dottedGeometry.drawDottedGeometry(dottedGeometryColor, GUIDottedGeometry::DottedContourType::INSPECT);
                }
                // pop matrix
                GLHelper::popMatrix();
            }
        }
    }
}


GNELane*
GNEEdgeRelData::getFirstPathLane() const {
    /* temporal */
    return nullptr;
}


GNELane*
GNEEdgeRelData::getLastPathLane() const {
    /* temporal */
    return nullptr;
}


Position
GNEEdgeRelData::getPositionInView() const {
    return getParentEdges().front()->getPositionInView();
}


void
GNEEdgeRelData::writeGenericData(OutputDevice& device) const {
    // open device
    device.openTag(SUMO_TAG_EDGEREL);
    // write from
    device.writeAttr(SUMO_ATTR_FROM, getParentEdges().front()->getID());
    // write to
    device.writeAttr(SUMO_ATTR_TO, getParentEdges().back()->getID());
    // iterate over attributes
    for (const auto& attribute : getParametersMap()) {
        // write attribute (don't use writeParams)
        device.writeAttr(attribute.first, attribute.second);
    }
    // close device
    device.closeTag();
}


bool
GNEEdgeRelData::isGenericDataValid() const {
    return true;
}


std::string
GNEEdgeRelData::getGenericDataProblem() const {
    return "";
}


void
GNEEdgeRelData::fixGenericDataProblem() {
    throw InvalidArgument(getTagStr() + " cannot fix any problem");
}


Boundary
GNEEdgeRelData::getCenteringBoundary() const {
    return getParentEdges().front()->getCenteringBoundary();
}


std::string
GNEEdgeRelData::getAttribute(SumoXMLAttr key) const {
    switch (key) {
        case SUMO_ATTR_ID:
            return getParentEdges().front()->getID();
        case SUMO_ATTR_FROM:
            return getParentEdges().front()->getID();
        case SUMO_ATTR_TO:
            return getParentEdges().back()->getID();
        case GNE_ATTR_DATASET:
            return myDataIntervalParent->getDataSetParent()->getID();
        case GNE_ATTR_SELECTED:
            return toString(isAttributeCarrierSelected());
        case GNE_ATTR_PARAMETERS:
            return getParametersStr();
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


double
GNEEdgeRelData::getAttributeDouble(SumoXMLAttr key) const {
    throw InvalidArgument(getTagStr() + " doesn't have a double attribute of type '" + toString(key) + "'");
}


void
GNEEdgeRelData::setAttribute(SumoXMLAttr key, const std::string& value, GNEUndoList* undoList) {
    if (value == getAttribute(key)) {
        return; //avoid needless changes, later logic relies on the fact that attributes have changed
    }
    switch (key) {
        case SUMO_ATTR_FROM:
        case SUMO_ATTR_TO:
        case GNE_ATTR_SELECTED:
        case GNE_ATTR_PARAMETERS:
            undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


bool
GNEEdgeRelData::isValid(SumoXMLAttr key, const std::string& value) {
    switch (key) {
        case SUMO_ATTR_FROM:
            return SUMOXMLDefinitions::isValidNetID(value) && (myNet->getAttributeCarriers()->retrieveEdge(value, false) != nullptr) &&
                   (value != getParentEdges().back()->getID());
        case SUMO_ATTR_TO:
            return SUMOXMLDefinitions::isValidNetID(value) && (myNet->getAttributeCarriers()->retrieveEdge(value, false) != nullptr) &&
                   (value != getParentEdges().front()->getID());
        case GNE_ATTR_SELECTED:
            return canParse<bool>(value);
        case GNE_ATTR_PARAMETERS:
            return Parameterised::areAttributesValid(value, true);
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEEdgeRelData::enableAttribute(SumoXMLAttr /*key*/, GNEUndoList* /*undoList*/) {
    // Nothing to enable
}


void
GNEEdgeRelData::disableAttribute(SumoXMLAttr /*key*/, GNEUndoList* /*undoList*/) {
    // Nothing to disable enable
}


bool GNEEdgeRelData::isAttributeEnabled(SumoXMLAttr key) const {
    switch (key) {
        case SUMO_ATTR_ID:
            return false;
        default:
            return true;
    }
}


std::string
GNEEdgeRelData::getPopUpID() const {
    return getTagStr();
}


std::string
GNEEdgeRelData::getHierarchyName() const {
    return getTagStr() + ": " + getParentEdges().front()->getID() + "->" + getParentEdges().back()->getID();
}


void
GNEEdgeRelData::setAttribute(SumoXMLAttr key, const std::string& value) {
    switch (key) {
        case SUMO_ATTR_FROM: {
            // change first edge
            replaceFirstParentEdge(value);
            break;
        }
        case SUMO_ATTR_TO: {
            // change last edge
            replaceLastParentEdge(value);
            break;
        }
        case GNE_ATTR_SELECTED:
            if (parse<bool>(value)) {
                selectAttributeCarrier();
            } else {
                unselectAttributeCarrier();
            }
            break;
        case GNE_ATTR_PARAMETERS:
            setParametersStr(value);
            // update attribute colors
            myDataIntervalParent->getDataSetParent()->updateAttributeColors();
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
    // mark interval toolbar for update
    myNet->getViewNet()->getIntervalBar().markForUpdate();
}


void
GNEEdgeRelData::toogleAttribute(SumoXMLAttr /*key*/, const bool /*value*/) {
    throw InvalidArgument("Nothing to enable");
}

/****************************************************************************/
