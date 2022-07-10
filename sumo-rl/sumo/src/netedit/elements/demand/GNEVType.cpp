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
/// @file    GNEVType.cpp
/// @author  Pablo Alvarez Lopez
/// @date    Jan 2019
///
// Definition of Vehicle Types in NETEDIT
/****************************************************************************/
#include <netedit/GNENet.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNEUndoList.h>
#include <netedit/changes/GNEChange_Attribute.h>
#include <netedit/changes/GNEChange_DemandElement.h>
#include <utils/emissions/PollutantsInterface.h>

#include "GNEVType.h"
#include "GNEVTypeDistribution.h"


// ===========================================================================
// member method definitions
// ===========================================================================

GNEVType::GNEVType(GNENet* net) :
    GNEDemandElement("", net, GLO_VTYPE, SUMO_TAG_VTYPE, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {}, {}, {}),
SUMOVTypeParameter(""),
myDefaultVehicleType(true),
myDefaultVehicleTypeModified(false) {
    // reset default values
    resetDefaultValues();
    // init Rail Visualization Parameters
    initRailVisualizationParameters();
}


GNEVType::GNEVType(GNENet* net, const std::string& vTypeID, const SUMOVehicleClass& defaultVClass) :
    GNEDemandElement(vTypeID, net, GLO_VTYPE, SUMO_TAG_VTYPE, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {}, {}, {}),
SUMOVTypeParameter(vTypeID),
myDefaultVehicleType(true),
myDefaultVehicleTypeModified(false) {
    // set default vehicle class
    vehicleClass = defaultVClass;
    // init Rail Visualization Parameters
    initRailVisualizationParameters();
}


GNEVType::GNEVType(GNENet* net, const SUMOVTypeParameter& vTypeParameter) :
    GNEDemandElement(vTypeParameter.id, net, GLO_VTYPE, SUMO_TAG_VTYPE, GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {}, {}, {}),
SUMOVTypeParameter(vTypeParameter),
myDefaultVehicleType(false),
myDefaultVehicleTypeModified(false) {
    // init Rail Visualization Parameters
    initRailVisualizationParameters();
}


GNEVType::GNEVType(GNENet* net, const std::string& vTypeID, GNEVType* vTypeOriginal) :
    GNEDemandElement(vTypeID, net, GLO_VTYPE, vTypeOriginal->getTagProperty().getTag(), GNEPathManager::PathElement::Options::DEMAND_ELEMENT,
{}, {}, {}, {}, {}, {}),
SUMOVTypeParameter(*vTypeOriginal),
myDefaultVehicleType(false),
myDefaultVehicleTypeModified(false) {
    // change manually the ID (to avoid to use the ID of vTypeOriginal)
    id = vTypeID;
    // init Rail Visualization Parameters
    initRailVisualizationParameters();
}


GNEVType::~GNEVType() {}


GNEMoveOperation*
GNEVType::getMoveOperation() {
    return nullptr;
}


void
GNEVType::writeDemandElement(OutputDevice& device) const {
    // only write default vehicle types if it was modified
    if (myDefaultVehicleType) {
        if (myDefaultVehicleTypeModified || (getParentDemandElements().size() > 0)) {
            write(device);
        }
    } else {
        write(device);
    }
}


GNEDemandElement::Problem
GNEVType::isDemandElementValid() const {
    // currently vTypes don't have problems
    return GNEDemandElement::Problem::OK;
}


std::string
GNEVType::getDemandElementProblem() const {
    return "";
}


void
GNEVType::fixDemandElementProblem() {
    // nothing to fix
}


SUMOVehicleClass
GNEVType::getVClass() const {
    return vehicleClass;
}


const RGBColor&
GNEVType::getColor() const {
    return color;
}


void
GNEVType::updateGeometry() {
    // update geometry of all childrens
    for (const auto& i : getChildDemandElements()) {
        i->updateGeometry();
    }
}


Position
GNEVType::getPositionInView() const {
    return Position();
}


std::string
GNEVType::getParentName() const {
    return myNet->getMicrosimID();
}


double
GNEVType::getExaggeration(const GUIVisualizationSettings& /*s*/) const {
    return 1;
}


Boundary
GNEVType::getCenteringBoundary() const {
    // Vehicle Types doesn't have boundaries
    return Boundary(-0.1, -0.1, 0.1, 0.1);
}


void
GNEVType::splitEdgeGeometry(const double /*splitPosition*/, const GNENetworkElement* /*originalElement*/, const GNENetworkElement* /*newElement*/, GNEUndoList* /*undoList*/) {
    // geometry of this element cannot be splitted
}


void
GNEVType::drawGL(const GUIVisualizationSettings&) const {
    // Vehicle Types aren't draw
}


void
GNEVType::computePathElement() {
    // nothing to compute
}


void
GNEVType::drawPartialGL(const GUIVisualizationSettings& /*s*/, const GNELane* /*lane*/, const GNEPathManager::Segment* /*segment*/, const double /*offsetFront*/) const {
    // vehicleTypes don't use drawPartialGL
}


void
GNEVType::drawPartialGL(const GUIVisualizationSettings& /*s*/, const GNELane* /*fromLane*/, const GNELane* /*toLane*/, const GNEPathManager::Segment* /*segment*/, const double /*offsetFront*/) const {
    // vehicleTypes don't use drawPartialGL
}


GNELane*
GNEVType::getFirstPathLane() const {
    // vehicle types don't use lanes
    return nullptr;
}


GNELane*
GNEVType::getLastPathLane() const {
    // vehicle types don't use lanes
    return nullptr;
}


std::string
GNEVType::getAttribute(SumoXMLAttr key) const {
    // obtain default values depending of vehicle class
    VClassDefaultValues defaultValues(vehicleClass);
    switch (key) {
        case SUMO_ATTR_ID:
            return getID();
        case GNE_ATTR_VTYPE_DISTRIBUTION:
            if (getParentDemandElements().empty()) {
                return "";
            } else {
                return getParentDemandElements().front()->getID();
            }
        // CFM Attributes
        case SUMO_ATTR_ACCEL:
        case SUMO_ATTR_DECEL:
        case SUMO_ATTR_APPARENTDECEL:
        case SUMO_ATTR_EMERGENCYDECEL:
        case SUMO_ATTR_SIGMA:
        case SUMO_ATTR_TAU:
            // this CFM has default values
            return getCFParamString(key, myTagProperty.getDefaultValue(key));
        // JM Attributes
        case SUMO_ATTR_JM_CROSSING_GAP:
        case SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME:
        case SUMO_ATTR_JM_DRIVE_RED_SPEED:
        case SUMO_ATTR_JM_IGNORE_FOE_PROB:
        case SUMO_ATTR_JM_IGNORE_FOE_SPEED:
        case SUMO_ATTR_JM_SIGMA_MINOR:
        case SUMO_ATTR_JM_TIMEGAP_MINOR:
            // this JM has default values
            return getJMParamString(key, myTagProperty.getDefaultValue(key));
        case SUMO_ATTR_IMPATIENCE:
            if (wasSet(VTYPEPARS_IMPATIENCE_SET)) {
                return toString(impatience);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_IMPATIENCE);
            }
        // LCM Attributes
        case SUMO_ATTR_LCA_STRATEGIC_PARAM:
        case SUMO_ATTR_LCA_COOPERATIVE_PARAM:
        case SUMO_ATTR_LCA_SPEEDGAIN_PARAM:
        case SUMO_ATTR_LCA_KEEPRIGHT_PARAM:
        case SUMO_ATTR_LCA_SUBLANE_PARAM:
        case SUMO_ATTR_LCA_OPPOSITE_PARAM:
        case SUMO_ATTR_LCA_PUSHY:
        case SUMO_ATTR_LCA_PUSHYGAP:
        case SUMO_ATTR_LCA_ASSERTIVE:
        case SUMO_ATTR_LCA_IMPATIENCE:
        case SUMO_ATTR_LCA_TIME_TO_IMPATIENCE:
        case SUMO_ATTR_LCA_ACCEL_LAT:
        case SUMO_ATTR_LCA_LOOKAHEADLEFT:
        case SUMO_ATTR_LCA_SPEEDGAINRIGHT:
        case SUMO_ATTR_LCA_MAXSPEEDLATSTANDING:
        case SUMO_ATTR_LCA_MAXSPEEDLATFACTOR:
        case SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE:
        case SUMO_ATTR_LCA_OVERTAKE_RIGHT:
            /* case SUMO_ATTR_LCA_EXPERIMENTAL1: */
            return getLCParamString(key, myTagProperty.getDefaultValue(key));
        //
        case SUMO_ATTR_COLLISION_MINGAP_FACTOR:
        case SUMO_ATTR_TMP1:
        case SUMO_ATTR_TMP2:
        case SUMO_ATTR_TMP3:
        case SUMO_ATTR_TMP4:
        case SUMO_ATTR_TMP5:
        case SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS:
        case SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW:
        case SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE:
        case SUMO_ATTR_CF_EIDM_T_REACTION:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE:
        case SUMO_ATTR_CF_EIDM_C_COOLNESS:
        case SUMO_ATTR_CF_EIDM_SIG_LEADER:
        case SUMO_ATTR_CF_EIDM_SIG_GAP:
        case SUMO_ATTR_CF_EIDM_SIG_ERROR:
        case SUMO_ATTR_CF_EIDM_JERK_MAX:
        case SUMO_ATTR_CF_EIDM_EPSILON_ACC:
        case SUMO_ATTR_CF_EIDM_T_ACC_MAX:
        case SUMO_ATTR_CF_EIDM_M_FLATNESS:
        case SUMO_ATTR_CF_EIDM_M_BEGIN:
        case SUMO_ATTR_CF_PWAGNER2009_TAULAST:
        case SUMO_ATTR_CF_PWAGNER2009_APPROB:
        case SUMO_ATTR_CF_IDMM_ADAPT_FACTOR:
        case SUMO_ATTR_CF_IDMM_ADAPT_TIME:
        case SUMO_ATTR_CF_WIEDEMANN_SECURITY:
        case SUMO_ATTR_CF_WIEDEMANN_ESTIMATION:
        case SUMO_ATTR_TRAIN_TYPE:
        case SUMO_ATTR_K:
        case SUMO_ATTR_CF_KERNER_PHI:
        case SUMO_ATTR_CF_IDM_DELTA:
        case SUMO_ATTR_CF_IDM_STEPPING:
            return getCFParamString(key, "");
        // Mutable values
        case SUMO_ATTR_LENGTH:
            if (wasSet(VTYPEPARS_LENGTH_SET)) {
                return toString(length);
            } else {
                return toString(defaultValues.length);
            }
        case SUMO_ATTR_MINGAP:
            if (wasSet(VTYPEPARS_MINGAP_SET)) {
                return toString(minGap);
            } else {
                return toString(defaultValues.minGap);
            }
        case SUMO_ATTR_MAXSPEED:
            if (wasSet(VTYPEPARS_MAXSPEED_SET)) {
                return toString(maxSpeed);
            } else {
                return toString(defaultValues.maxSpeed);
            }
        case SUMO_ATTR_SPEEDFACTOR:
            if (wasSet(VTYPEPARS_SPEEDFACTOR_SET)) {
                return toString(speedFactor);
            } else {
                return toString(defaultValues.speedFactor);
            }
        case SUMO_ATTR_PERSON_CAPACITY:
            if (wasSet(VTYPEPARS_PERSON_CAPACITY)) {
                return toString(personCapacity);
            } else {
                return toString(defaultValues.personCapacity);
            }
        case SUMO_ATTR_CONTAINER_CAPACITY:
            if (wasSet(VTYPEPARS_CONTAINER_CAPACITY)) {
                return toString(containerCapacity);
            } else {
                return toString(defaultValues.containerCapacity);
            }
        case SUMO_ATTR_OSGFILE:
            if (wasSet(VTYPEPARS_OSGFILE_SET)) {
                return osgFile;
            } else {
                return defaultValues.osgFile;
            }
        case SUMO_ATTR_COLOR:
            if (wasSet(VTYPEPARS_COLOR_SET)) {
                return toString(color);
            } else {
                return "";
            }
        case SUMO_ATTR_VCLASS:
            if (wasSet(VTYPEPARS_VEHICLECLASS_SET)) {
                return toString(vehicleClass);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_VCLASS);
            }
        case SUMO_ATTR_EMISSIONCLASS:
            if (wasSet(VTYPEPARS_EMISSIONCLASS_SET)) {
                return PollutantsInterface::getName(emissionClass);
            } else {
                return PollutantsInterface::getName(defaultValues.emissionClass);
            }
        case SUMO_ATTR_GUISHAPE:
            if (wasSet(VTYPEPARS_SHAPE_SET)) {
                return getVehicleShapeName(shape);
            } else {
                return getVehicleShapeName(defaultValues.shape);
            }
        case SUMO_ATTR_WIDTH:
            if (wasSet(VTYPEPARS_WIDTH_SET)) {
                return toString(width);
            } else {
                return toString(defaultValues.width);
            }
        case SUMO_ATTR_HEIGHT:
            if (wasSet(VTYPEPARS_HEIGHT_SET)) {
                return toString(height);
            } else {
                return toString(defaultValues.height);
            }
        case SUMO_ATTR_IMGFILE:
            if (wasSet(VTYPEPARS_IMGFILE_SET)) {
                return imgFile;
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_IMGFILE);
            }
        case SUMO_ATTR_LANE_CHANGE_MODEL:
            if (wasSet(VTYPEPARS_LANE_CHANGE_MODEL_SET)) {
                return SUMOXMLDefinitions::LaneChangeModels.getString(lcModel);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_LANE_CHANGE_MODEL);
            }
        case SUMO_ATTR_CAR_FOLLOW_MODEL:
            if (wasSet(VTYPEPARS_CAR_FOLLOW_MODEL)) {
                return SUMOXMLDefinitions::CarFollowModels.getString(cfModel);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_CAR_FOLLOW_MODEL);
            }
        case SUMO_ATTR_BOARDING_DURATION:
            if (wasSet(VTYPEPARS_BOARDING_DURATION)) {
                return time2string(boardingDuration);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_BOARDING_DURATION);
            }
        case SUMO_ATTR_LOADING_DURATION:
            if (wasSet(VTYPEPARS_LOADING_DURATION)) {
                return time2string(loadingDuration);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_LOADING_DURATION);
            }
        case SUMO_ATTR_LATALIGNMENT:
            if (wasSet(VTYPEPARS_LATALIGNMENT_SET)) {
                if (latAlignmentProcedure != LatAlignmentDefinition::GIVEN) {
                    return toString(latAlignmentProcedure);
                } else {
                    return toString(latAlignmentOffset);
                }
            } else {
                return toString(defaultValues.latAlignmentProcedure);
            }
        case SUMO_ATTR_MINGAP_LAT:
            if (wasSet(VTYPEPARS_MINGAP_LAT_SET)) {
                return toString(minGapLat);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_MINGAP_LAT);
            }
        case SUMO_ATTR_MAXSPEED_LAT:
            if (wasSet(VTYPEPARS_MAXSPEED_LAT_SET)) {
                return toString(maxSpeedLat);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_MAXSPEED_LAT);
            }
        case SUMO_ATTR_ACTIONSTEPLENGTH:
            if (wasSet(VTYPEPARS_ACTIONSTEPLENGTH_SET)) {
                return toString(actionStepLength);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_ACTIONSTEPLENGTH);
            }
        case SUMO_ATTR_PROB:
            if (wasSet(VTYPEPARS_PROBABILITY_SET)) {
                return toString(defaultProbability);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_PROB);
            }
        case SUMO_ATTR_CARRIAGE_LENGTH:
            if (wasSet(VTYPEPARS_CARRIAGE_LENGTH_SET)) {
                return toString(carriageLength);
            } else {
                return toString(defaultValues.carriageLength);
            }
        case SUMO_ATTR_LOCOMOTIVE_LENGTH:
            if (wasSet(VTYPEPARS_LOCOMOTIVE_LENGTH_SET)) {
                return toString(locomotiveLength);
            } else {
                return toString(defaultValues.locomotiveLength);
            }
        case SUMO_ATTR_CARRIAGE_GAP:
            if (wasSet(VTYPEPARS_CARRIAGE_GAP_SET)) {
                return toString(carriageGap);
            } else {
                return myTagProperty.getDefaultValue(SUMO_ATTR_CARRIAGE_GAP);
            }
        case GNE_ATTR_SELECTED:
            return toString(isAttributeCarrierSelected());
        case GNE_ATTR_PARAMETERS:
            return getParametersStr();
        // other
        case GNE_ATTR_DEFAULT_VTYPE:
            return toString(myDefaultVehicleType);
        case GNE_ATTR_DEFAULT_VTYPE_MODIFIED:
            if (myDefaultVehicleType) {
                return toString(myDefaultVehicleTypeModified);
            } else {
                return False;
            }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


double
GNEVType::getAttributeDouble(SumoXMLAttr key) const {
    // obtain default values depending of vehicle class
    VClassDefaultValues defaultValues(vehicleClass);
    switch (key) {
        case SUMO_ATTR_LENGTH:
            if (wasSet(VTYPEPARS_LENGTH_SET)) {
                return length;
            } else {
                return defaultValues.length;
            }
        case SUMO_ATTR_MINGAP:
            if (wasSet(VTYPEPARS_MINGAP_SET)) {
                return minGap;
            } else {
                return defaultValues.minGap;
            }
        case SUMO_ATTR_WIDTH:
            if (wasSet(VTYPEPARS_WIDTH_SET)) {
                return width;
            } else {
                return defaultValues.width;
            }
        case SUMO_ATTR_HEIGHT:
            if (wasSet(VTYPEPARS_HEIGHT_SET)) {
                return height;
            } else {
                return defaultValues.height;
            }
        case SUMO_ATTR_MAXSPEED:
            if (wasSet(VTYPEPARS_MAXSPEED_SET)) {
                return maxSpeed;
            } else {
                return defaultValues.maxSpeed;
            }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have a double attribute of type '" + toString(key) + "'");
    }
}


Position
GNEVType::getAttributePosition(SumoXMLAttr key) const {
    throw InvalidArgument(getTagStr() + " doesn't have a Position attribute of type '" + toString(key) + "'");
}


void
GNEVType::setAttribute(SumoXMLAttr key, const std::string& value, GNEUndoList* undoList) {
    GNEChange_Attribute* vTypeChangeAttributeForced = nullptr;
    if (value == getAttribute(key)) {
        return; //avoid needless changes, later logic relies on the fact that attributes have changed
    }
    switch (key) {
        case SUMO_ATTR_ID:
            undoList->changeAttribute(new GNEChange_Attribute(this, key, value));
            break;
        case GNE_ATTR_VTYPE_DISTRIBUTION:
            editVTypeDistribution(value, undoList);
            break;
        // CFM Attributes
        case SUMO_ATTR_ACCEL:
        case SUMO_ATTR_DECEL:
        case SUMO_ATTR_APPARENTDECEL:
        case SUMO_ATTR_EMERGENCYDECEL:
        case SUMO_ATTR_SIGMA:
        case SUMO_ATTR_TAU:
        case SUMO_ATTR_COLLISION_MINGAP_FACTOR:
        case SUMO_ATTR_TMP1:
        case SUMO_ATTR_TMP2:
        case SUMO_ATTR_TMP3:
        case SUMO_ATTR_TMP4:
        case SUMO_ATTR_TMP5:
        case SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS:
        case SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW:
        case SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE:
        case SUMO_ATTR_CF_EIDM_T_REACTION:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE:
        case SUMO_ATTR_CF_EIDM_C_COOLNESS:
        case SUMO_ATTR_CF_EIDM_SIG_LEADER:
        case SUMO_ATTR_CF_EIDM_SIG_GAP:
        case SUMO_ATTR_CF_EIDM_SIG_ERROR:
        case SUMO_ATTR_CF_EIDM_JERK_MAX:
        case SUMO_ATTR_CF_EIDM_EPSILON_ACC:
        case SUMO_ATTR_CF_EIDM_T_ACC_MAX:
        case SUMO_ATTR_CF_EIDM_M_FLATNESS:
        case SUMO_ATTR_CF_EIDM_M_BEGIN:
        case SUMO_ATTR_CF_PWAGNER2009_TAULAST:
        case SUMO_ATTR_CF_PWAGNER2009_APPROB:
        case SUMO_ATTR_CF_IDMM_ADAPT_FACTOR:
        case SUMO_ATTR_CF_IDMM_ADAPT_TIME:
        case SUMO_ATTR_CF_WIEDEMANN_SECURITY:
        case SUMO_ATTR_CF_WIEDEMANN_ESTIMATION:
        case SUMO_ATTR_TRAIN_TYPE:
        case SUMO_ATTR_K:
        case SUMO_ATTR_CF_KERNER_PHI:
        case SUMO_ATTR_CF_IDM_DELTA:
        case SUMO_ATTR_CF_IDM_STEPPING:
        // JM Attributes
        case SUMO_ATTR_JM_CROSSING_GAP:
        case SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME:
        case SUMO_ATTR_JM_DRIVE_RED_SPEED:
        case SUMO_ATTR_JM_IGNORE_FOE_PROB:
        case SUMO_ATTR_JM_IGNORE_FOE_SPEED:
        case SUMO_ATTR_JM_SIGMA_MINOR:
        case SUMO_ATTR_JM_TIMEGAP_MINOR:
        case SUMO_ATTR_IMPATIENCE:
        // LCM Attributes
        case SUMO_ATTR_LCA_STRATEGIC_PARAM:
        case SUMO_ATTR_LCA_COOPERATIVE_PARAM:
        case SUMO_ATTR_LCA_SPEEDGAIN_PARAM:
        case SUMO_ATTR_LCA_KEEPRIGHT_PARAM:
        case SUMO_ATTR_LCA_SUBLANE_PARAM:
        case SUMO_ATTR_LCA_OPPOSITE_PARAM:
        case SUMO_ATTR_LCA_PUSHY:
        case SUMO_ATTR_LCA_PUSHYGAP:
        case SUMO_ATTR_LCA_ASSERTIVE:
        case SUMO_ATTR_LCA_IMPATIENCE:
        case SUMO_ATTR_LCA_TIME_TO_IMPATIENCE:
        case SUMO_ATTR_LCA_ACCEL_LAT:
        case SUMO_ATTR_LCA_LOOKAHEADLEFT:
        case SUMO_ATTR_LCA_SPEEDGAINRIGHT:
        case SUMO_ATTR_LCA_MAXSPEEDLATSTANDING:
        case SUMO_ATTR_LCA_MAXSPEEDLATFACTOR:
        case SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE:
        case SUMO_ATTR_LCA_OVERTAKE_RIGHT:
        /* case SUMO_ATTR_LCA_EXPERIMENTAL1: */
        //
        case SUMO_ATTR_LENGTH:
        case SUMO_ATTR_MINGAP:
        case SUMO_ATTR_MAXSPEED:
        case SUMO_ATTR_SPEEDFACTOR:
        case SUMO_ATTR_COLOR:
        case SUMO_ATTR_VCLASS:
        case SUMO_ATTR_EMISSIONCLASS:
        case SUMO_ATTR_GUISHAPE:
        case SUMO_ATTR_WIDTH:
        case SUMO_ATTR_HEIGHT:
        case SUMO_ATTR_IMGFILE:
        case SUMO_ATTR_LANE_CHANGE_MODEL:
        case SUMO_ATTR_CAR_FOLLOW_MODEL:
        case SUMO_ATTR_PERSON_CAPACITY:
        case SUMO_ATTR_CONTAINER_CAPACITY:
        case SUMO_ATTR_BOARDING_DURATION:
        case SUMO_ATTR_LOADING_DURATION:
        case SUMO_ATTR_LATALIGNMENT:
        case SUMO_ATTR_MINGAP_LAT:
        case SUMO_ATTR_MAXSPEED_LAT:
        case SUMO_ATTR_ACTIONSTEPLENGTH:
        case SUMO_ATTR_PROB:
        case SUMO_ATTR_OSGFILE:
        case SUMO_ATTR_CARRIAGE_LENGTH:
        case SUMO_ATTR_LOCOMOTIVE_LENGTH:
        case SUMO_ATTR_CARRIAGE_GAP:
        case GNE_ATTR_SELECTED:
        case GNE_ATTR_PARAMETERS:
            // if we change the original value of a default vehicle Type, change also flag "myDefaultVehicleType"
            if (myDefaultVehicleType) {
                vTypeChangeAttributeForced = new GNEChange_Attribute(this, GNE_ATTR_DEFAULT_VTYPE_MODIFIED, "true");
                // force change
                vTypeChangeAttributeForced->forceChange();
                undoList->changeAttribute(vTypeChangeAttributeForced);
            }
            vTypeChangeAttributeForced = new GNEChange_Attribute(this, key, value);
            // force change
            vTypeChangeAttributeForced->forceChange();
            undoList->changeAttribute(vTypeChangeAttributeForced);
            break;
        case GNE_ATTR_DEFAULT_VTYPE_MODIFIED:
            vTypeChangeAttributeForced = new GNEChange_Attribute(this, GNE_ATTR_DEFAULT_VTYPE_MODIFIED, "true");
            // force change
            vTypeChangeAttributeForced->forceChange();
            undoList->changeAttribute(vTypeChangeAttributeForced);
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


bool
GNEVType::isValid(SumoXMLAttr key, const std::string& value) {
    // a empty value is always valid except for IDs
    if ((key != SUMO_ATTR_ID) && value.empty()) {
        return true;
    }
    switch (key) {
        case SUMO_ATTR_ID:
            // Vtypes and PTypes shares namespace
            if (value == getID()) {
                return true;
            } else if (SUMOXMLDefinitions::isValidVehicleID(value) && (myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_VTYPE, value, false) == nullptr)) {
                return true;
            } else {
                return false;
            }
        case GNE_ATTR_VTYPE_DISTRIBUTION:
            if (value.empty()) {
                return true;
            } else {
                return SUMOXMLDefinitions::isValidVehicleID(value);
            }
        // CFM Attributes
        case SUMO_ATTR_SIGMA:
            return canParse<double>(value) && (parse<double>(value) >= 0) && (parse<double>(value) <= 1);
        case SUMO_ATTR_ACCEL:
        case SUMO_ATTR_DECEL:
        case SUMO_ATTR_APPARENTDECEL:
        case SUMO_ATTR_EMERGENCYDECEL:
        case SUMO_ATTR_TAU:
        case SUMO_ATTR_COLLISION_MINGAP_FACTOR:
        case SUMO_ATTR_TMP1:
        case SUMO_ATTR_TMP2:
        case SUMO_ATTR_TMP3:
        case SUMO_ATTR_TMP4:
        case SUMO_ATTR_TMP5:
        case SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS:
        case SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW:
        case SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE:
        case SUMO_ATTR_CF_EIDM_T_REACTION:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE:
        case SUMO_ATTR_CF_EIDM_C_COOLNESS:
        case SUMO_ATTR_CF_EIDM_SIG_LEADER:
        case SUMO_ATTR_CF_EIDM_SIG_GAP:
        case SUMO_ATTR_CF_EIDM_SIG_ERROR:
        case SUMO_ATTR_CF_EIDM_JERK_MAX:
        case SUMO_ATTR_CF_EIDM_EPSILON_ACC:
        case SUMO_ATTR_CF_EIDM_T_ACC_MAX:
        case SUMO_ATTR_CF_EIDM_M_FLATNESS:
        case SUMO_ATTR_CF_EIDM_M_BEGIN:
        case SUMO_ATTR_CF_PWAGNER2009_TAULAST:
        case SUMO_ATTR_CF_PWAGNER2009_APPROB:
        case SUMO_ATTR_CF_IDMM_ADAPT_FACTOR:
        case SUMO_ATTR_CF_IDMM_ADAPT_TIME:
        case SUMO_ATTR_CF_WIEDEMANN_SECURITY:
        case SUMO_ATTR_CF_WIEDEMANN_ESTIMATION:
        case SUMO_ATTR_K:
        case SUMO_ATTR_CF_KERNER_PHI:
        case SUMO_ATTR_CF_IDM_DELTA:
        case SUMO_ATTR_CF_IDM_STEPPING:
            return canParse<double>(value);
        case SUMO_ATTR_TRAIN_TYPE:
            // rail string
            return SUMOXMLDefinitions::TrainTypes.hasString(value);
        // JM Attributes
        case SUMO_ATTR_JM_CROSSING_GAP:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME:
            return canParse<double>(value) && (parse<double>(value) >= -1);
        case SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME:
            return canParse<double>(value) && (parse<double>(value) >= -1);
        case SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME:
            return canParse<double>(value) && (parse<double>(value) >= -1);
        case SUMO_ATTR_JM_DRIVE_RED_SPEED:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_JM_IGNORE_FOE_PROB:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_JM_IGNORE_FOE_SPEED:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_JM_SIGMA_MINOR:
            return canParse<double>(value) && (parse<double>(value) >= 0) && (parse<double>(value) <= 1);
        case SUMO_ATTR_JM_TIMEGAP_MINOR:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_IMPATIENCE:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        // LCM Attributes
        case SUMO_ATTR_LCA_STRATEGIC_PARAM:
        case SUMO_ATTR_LCA_COOPERATIVE_PARAM:
        case SUMO_ATTR_LCA_SPEEDGAIN_PARAM:
        case SUMO_ATTR_LCA_KEEPRIGHT_PARAM:
        case SUMO_ATTR_LCA_SUBLANE_PARAM:
        case SUMO_ATTR_LCA_OPPOSITE_PARAM:
        case SUMO_ATTR_LCA_PUSHY:
            return canParse<double>(value);
        case SUMO_ATTR_LCA_PUSHYGAP:
        case SUMO_ATTR_LCA_IMPATIENCE:
        case SUMO_ATTR_LCA_MAXSPEEDLATSTANDING:
        case SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE: // 0 mean disabled
        case SUMO_ATTR_LCA_TIME_TO_IMPATIENCE:      // 0 mean disabled
        case SUMO_ATTR_LCA_OVERTAKE_RIGHT:          // 0 mean disabled
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_LCA_ASSERTIVE:
        case SUMO_ATTR_LCA_LOOKAHEADLEFT:
        case SUMO_ATTR_LCA_SPEEDGAINRIGHT:
        case SUMO_ATTR_LCA_ACCEL_LAT:
        case SUMO_ATTR_LCA_MAXSPEEDLATFACTOR:
            return canParse<double>(value) && (parse<double>(value) > 0);
        /* case SUMO_ATTR_LCA_EXPERIMENTAL1:
            return true;
        */
        //
        case SUMO_ATTR_LENGTH:
            return canParse<double>(value) && (parse<double>(value) > 0);
        case SUMO_ATTR_MINGAP:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_MAXSPEED:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_SPEEDFACTOR:
            return Distribution_Parameterized::isValidDescription(value);
        case SUMO_ATTR_COLOR:
            if (value.empty()) {
                return true;
            } else {
                return canParse<RGBColor>(value);
            }
        case SUMO_ATTR_VCLASS:
            return canParseVehicleClasses(value);
        case SUMO_ATTR_EMISSIONCLASS:
            // check if given value correspond to a string of PollutantsInterface::getAllClassesStr()
            for (const auto& eClass : PollutantsInterface::getAllClassesStr()) {
                if (value == eClass) {
                    return true;
                }
            }
            return false;
        case SUMO_ATTR_GUISHAPE:
            if (value == "all") {
                return false;
            } else {
                return canParseVehicleShape(value);
            }
        case SUMO_ATTR_WIDTH:
            return canParse<double>(value);
        case SUMO_ATTR_HEIGHT:
            return canParse<double>(value);
        case SUMO_ATTR_IMGFILE:
            return SUMOXMLDefinitions::isValidFilename(value);
        case SUMO_ATTR_LANE_CHANGE_MODEL:
            return SUMOXMLDefinitions::LaneChangeModels.hasString(value);
        case SUMO_ATTR_CAR_FOLLOW_MODEL:
            return SUMOXMLDefinitions::CarFollowModels.hasString(value);
        case SUMO_ATTR_PERSON_CAPACITY:
            return canParse<int>(value);
        case SUMO_ATTR_CONTAINER_CAPACITY:
            return canParse<int>(value);
        case SUMO_ATTR_BOARDING_DURATION:
            return canParse<double>(value);
        case SUMO_ATTR_LOADING_DURATION:
            return canParse<double>(value);
        case SUMO_ATTR_LATALIGNMENT:
            return SUMOVTypeParameter::isValidLatAlignment(value);
        case SUMO_ATTR_MINGAP_LAT:
            return canParse<double>(value);
        case SUMO_ATTR_MAXSPEED_LAT:
            return canParse<double>(value);
        case SUMO_ATTR_ACTIONSTEPLENGTH:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_PROB:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case SUMO_ATTR_OSGFILE:
            return SUMOXMLDefinitions::isValidFilename(value);
        case SUMO_ATTR_CARRIAGE_LENGTH:
            return canParse<double>(value) && (parse<double>(value) >= -1);
        case SUMO_ATTR_LOCOMOTIVE_LENGTH:
            return canParse<double>(value) && (parse<double>(value) >= -1);
        case SUMO_ATTR_CARRIAGE_GAP:
            return canParse<double>(value) && (parse<double>(value) >= 0);
        case GNE_ATTR_SELECTED:
            return canParse<bool>(value);
        case GNE_ATTR_PARAMETERS:
            return Parameterised::areParametersValid(value);
        case GNE_ATTR_DEFAULT_VTYPE_MODIFIED:
            if (myDefaultVehicleType) {
                return canParse<bool>(value);
            } else {
                return false;
            }
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
}


void
GNEVType::enableAttribute(SumoXMLAttr /*key*/, GNEUndoList* /*undoList*/) {
    // nothing to enable
}


void
GNEVType::disableAttribute(SumoXMLAttr /*key*/, GNEUndoList* /*undoList*/) {
    // nothing to disable
}


bool
GNEVType::isAttributeEnabled(SumoXMLAttr key) const {
    switch (key) {
        case SUMO_ATTR_ID:
            if ((id == DEFAULT_VTYPE_ID) || (id == DEFAULT_PEDTYPE_ID) || (id == DEFAULT_BIKETYPE_ID)) {
                return false;
            } else {
                return true;
            }
        case SUMO_ATTR_LENGTH:
            return wasSet(VTYPEPARS_LENGTH_SET);
        case SUMO_ATTR_MINGAP:
            return wasSet(VTYPEPARS_MINGAP_SET);
        case SUMO_ATTR_MAXSPEED:
            return wasSet(VTYPEPARS_MAXSPEED_SET);
        case SUMO_ATTR_SPEEDFACTOR:
            return wasSet(VTYPEPARS_SPEEDFACTOR_SET);
        case SUMO_ATTR_PERSON_CAPACITY:
            return wasSet(VTYPEPARS_PERSON_CAPACITY);
        case SUMO_ATTR_CONTAINER_CAPACITY:
            return wasSet(VTYPEPARS_CONTAINER_CAPACITY);
        case SUMO_ATTR_OSGFILE:
            return wasSet(VTYPEPARS_OSGFILE_SET);
        case SUMO_ATTR_CARRIAGE_LENGTH:
            return wasSet(VTYPEPARS_CARRIAGE_LENGTH_SET);
        case SUMO_ATTR_LOCOMOTIVE_LENGTH:
            return wasSet(VTYPEPARS_LOCOMOTIVE_LENGTH_SET);
        case SUMO_ATTR_CARRIAGE_GAP:
            return wasSet(VTYPEPARS_CARRIAGE_GAP_SET);
        default:
            return true;
    }
}


std::string
GNEVType::getPopUpID() const {
    return getTagStr();
}


std::string
GNEVType::getHierarchyName() const {
    return getTagStr() + ": " + getAttribute(SUMO_ATTR_ID) ;
}


const Parameterised::Map&
GNEVType::getACParametersMap() const {
    return getParametersMap();
}


void
GNEVType::overwriteVType(GNEDemandElement* vType, const SUMOVTypeParameter newVTypeParameter, GNEUndoList* undoList) {
    // open undo list and overwrite all values of default VType
    undoList->begin(vType->getTagProperty().getGUIIcon(), "update default " + vType->getTagStr() + " '" + DEFAULT_VTYPE_ID + "'");
    // CFM values
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_ACCEL, "").empty()) {
        vType->setAttribute(SUMO_ATTR_ACCEL, toString(newVTypeParameter.getCFParam(SUMO_ATTR_ACCEL, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_DECEL, "").empty()) {
        vType->setAttribute(SUMO_ATTR_DECEL, toString(newVTypeParameter.getCFParam(SUMO_ATTR_DECEL, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_APPARENTDECEL, "").empty()) {
        vType->setAttribute(SUMO_ATTR_APPARENTDECEL, toString(newVTypeParameter.getCFParam(SUMO_ATTR_APPARENTDECEL, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_EMERGENCYDECEL, "").empty()) {
        vType->setAttribute(SUMO_ATTR_EMERGENCYDECEL, toString(newVTypeParameter.getCFParam(SUMO_ATTR_EMERGENCYDECEL, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_SIGMA, "").empty()) {
        vType->setAttribute(SUMO_ATTR_SIGMA, toString(newVTypeParameter.getCFParam(SUMO_ATTR_SIGMA, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TAU, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TAU, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TAU, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_COLLISION_MINGAP_FACTOR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_COLLISION_MINGAP_FACTOR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_COLLISION_MINGAP_FACTOR, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TMP1, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TMP1, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TMP1, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TMP2, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TMP2, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TMP2, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TMP3, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TMP3, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TMP3, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TMP4, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TMP4, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TMP4, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TMP5, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TMP5, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TMP5, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_T_REACTION, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_T_REACTION, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_T_REACTION, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_C_COOLNESS, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_C_COOLNESS, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_C_COOLNESS, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_SIG_LEADER, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_SIG_LEADER, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_SIG_LEADER, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_SIG_GAP, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_SIG_GAP, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_SIG_GAP, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_SIG_ERROR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_SIG_ERROR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_SIG_ERROR, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_JERK_MAX, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_JERK_MAX, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_JERK_MAX, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_EPSILON_ACC, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_EPSILON_ACC, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_EPSILON_ACC, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_T_ACC_MAX, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_T_ACC_MAX, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_T_ACC_MAX, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_M_FLATNESS, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_M_FLATNESS, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_M_FLATNESS, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_EIDM_M_BEGIN, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_EIDM_M_BEGIN, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_EIDM_M_BEGIN, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_PWAGNER2009_TAULAST, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_PWAGNER2009_TAULAST, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_PWAGNER2009_TAULAST, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_PWAGNER2009_APPROB, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_PWAGNER2009_APPROB, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_PWAGNER2009_APPROB, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_IDMM_ADAPT_FACTOR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_IDMM_ADAPT_FACTOR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_IDMM_ADAPT_FACTOR, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_IDMM_ADAPT_TIME, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_IDMM_ADAPT_TIME, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_IDMM_ADAPT_TIME, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_WIEDEMANN_SECURITY, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_WIEDEMANN_SECURITY, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_WIEDEMANN_SECURITY, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_WIEDEMANN_ESTIMATION, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_WIEDEMANN_ESTIMATION, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_WIEDEMANN_ESTIMATION, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_TRAIN_TYPE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_TRAIN_TYPE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_TRAIN_TYPE, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_K, "").empty()) {
        vType->setAttribute(SUMO_ATTR_K, toString(newVTypeParameter.getCFParam(SUMO_ATTR_K, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_KERNER_PHI, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_KERNER_PHI, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_KERNER_PHI, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_IDM_DELTA, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_IDM_DELTA, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_IDM_DELTA, 0)), undoList);
    }
    if (!newVTypeParameter.getCFParamString(SUMO_ATTR_CF_IDM_STEPPING, "").empty()) {
        vType->setAttribute(SUMO_ATTR_CF_IDM_STEPPING, toString(newVTypeParameter.getCFParam(SUMO_ATTR_CF_IDM_STEPPING, 0)), undoList);
    }
    // JM values
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_CROSSING_GAP, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_CROSSING_GAP, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_CROSSING_GAP, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_DRIVE_RED_SPEED, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_DRIVE_RED_SPEED, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_DRIVE_RED_SPEED, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_IGNORE_FOE_PROB, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_IGNORE_FOE_PROB, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_IGNORE_FOE_PROB, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_IGNORE_FOE_SPEED, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_IGNORE_FOE_SPEED, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_IGNORE_FOE_SPEED, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_SIGMA_MINOR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_SIGMA_MINOR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_SIGMA_MINOR, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_JM_TIMEGAP_MINOR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_JM_TIMEGAP_MINOR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_JM_TIMEGAP_MINOR, 0)), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_IMPATIENCE_SET)) {
        vType->setAttribute(SUMO_ATTR_IMPATIENCE, toString(newVTypeParameter.impatience), undoList);
    }
    // LCM values
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_STRATEGIC_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_STRATEGIC_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_STRATEGIC_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_COOPERATIVE_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_COOPERATIVE_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_COOPERATIVE_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_SPEEDGAIN_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_SPEEDGAIN_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_SPEEDGAIN_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_KEEPRIGHT_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_KEEPRIGHT_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_KEEPRIGHT_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_SUBLANE_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_SUBLANE_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_SUBLANE_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_OPPOSITE_PARAM, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_OPPOSITE_PARAM, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_OPPOSITE_PARAM, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_PUSHY, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_PUSHY, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_PUSHY, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_PUSHYGAP, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_PUSHYGAP, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_PUSHYGAP, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_ASSERTIVE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_ASSERTIVE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_ASSERTIVE, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_IMPATIENCE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_IMPATIENCE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_IMPATIENCE, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_TIME_TO_IMPATIENCE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_TIME_TO_IMPATIENCE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_TIME_TO_IMPATIENCE, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_ACCEL_LAT, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_ACCEL_LAT, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_ACCEL_LAT, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_LOOKAHEADLEFT, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_LOOKAHEADLEFT, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_LOOKAHEADLEFT, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_SPEEDGAINRIGHT, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_SPEEDGAINRIGHT, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_SPEEDGAINRIGHT, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_MAXSPEEDLATSTANDING, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_MAXSPEEDLATSTANDING, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_MAXSPEEDLATSTANDING, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_MAXSPEEDLATFACTOR, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_MAXSPEEDLATFACTOR, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_MAXSPEEDLATFACTOR, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_OVERTAKE_RIGHT, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_OVERTAKE_RIGHT, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_OVERTAKE_RIGHT, 0)), undoList);
    }
    if (!newVTypeParameter.getJMParamString(SUMO_ATTR_LCA_EXPERIMENTAL1, "").empty()) {
        vType->setAttribute(SUMO_ATTR_LCA_EXPERIMENTAL1, toString(newVTypeParameter.getCFParam(SUMO_ATTR_LCA_EXPERIMENTAL1, 0)), undoList);
    }
    //
    if (newVTypeParameter.wasSet(VTYPEPARS_LENGTH_SET)) {
        vType->setAttribute(SUMO_ATTR_LENGTH, toString(newVTypeParameter.length), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_MINGAP_SET)) {
        vType->setAttribute(SUMO_ATTR_MINGAP, toString(newVTypeParameter.minGap), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_MAXSPEED_SET)) {
        vType->setAttribute(SUMO_ATTR_MAXSPEED, toString(newVTypeParameter.maxSpeed), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_SPEEDFACTOR_SET)) {
        vType->setAttribute(SUMO_ATTR_SPEEDFACTOR, toString(newVTypeParameter.speedFactor), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_COLOR_SET)) {
        vType->setAttribute(SUMO_ATTR_COLOR, toString(newVTypeParameter.color), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_EMISSIONCLASS_SET)) {
        vType->setAttribute(SUMO_ATTR_EMISSIONCLASS, PollutantsInterface::getName(newVTypeParameter.emissionClass), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_SHAPE_SET)) {
        vType->setAttribute(SUMO_ATTR_GUISHAPE, getVehicleShapeName(newVTypeParameter.shape), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_WIDTH_SET)) {
        vType->setAttribute(SUMO_ATTR_WIDTH, toString(newVTypeParameter.width), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_HEIGHT_SET)) {
        vType->setAttribute(SUMO_ATTR_HEIGHT, toString(newVTypeParameter.height), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_IMGFILE_SET)) {
        vType->setAttribute(SUMO_ATTR_IMGFILE, toString(newVTypeParameter.imgFile), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_LANE_CHANGE_MODEL_SET)) {
        vType->setAttribute(SUMO_ATTR_LANE_CHANGE_MODEL, SUMOXMLDefinitions::LaneChangeModels.getString(newVTypeParameter.lcModel), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_CAR_FOLLOW_MODEL)) {
        vType->setAttribute(SUMO_ATTR_CAR_FOLLOW_MODEL, SUMOXMLDefinitions::CarFollowModels.getString(newVTypeParameter.cfModel), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_PERSON_CAPACITY)) {
        vType->setAttribute(SUMO_ATTR_PERSON_CAPACITY, toString(newVTypeParameter.personCapacity), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_CONTAINER_CAPACITY)) {
        vType->setAttribute(SUMO_ATTR_CONTAINER_CAPACITY, toString(newVTypeParameter.containerCapacity), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_BOARDING_DURATION)) {
        vType->setAttribute(SUMO_ATTR_BOARDING_DURATION, toString(newVTypeParameter.boardingDuration), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_LOADING_DURATION)) {
        vType->setAttribute(SUMO_ATTR_LOADING_DURATION, toString(newVTypeParameter.loadingDuration), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_LATALIGNMENT_SET)) {
        if (newVTypeParameter.latAlignmentProcedure != LatAlignmentDefinition::GIVEN) {
            vType->setAttribute(SUMO_ATTR_LATALIGNMENT, toString(newVTypeParameter.latAlignmentProcedure), undoList);
        } else {
            vType->setAttribute(SUMO_ATTR_LATALIGNMENT, toString(newVTypeParameter.latAlignmentOffset), undoList);
        }
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_MINGAP_LAT_SET)) {
        vType->setAttribute(SUMO_ATTR_MINGAP_LAT, toString(newVTypeParameter.minGapLat), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_MAXSPEED_LAT_SET)) {
        vType->setAttribute(SUMO_ATTR_MAXSPEED_LAT, toString(newVTypeParameter.maxSpeedLat), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_ACTIONSTEPLENGTH_SET)) {
        vType->setAttribute(SUMO_ATTR_ACTIONSTEPLENGTH, toString(newVTypeParameter.actionStepLength), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_PROBABILITY_SET)) {
        vType->setAttribute(SUMO_ATTR_PROB, toString(newVTypeParameter.defaultProbability), undoList);
    }
    if (newVTypeParameter.wasSet(VTYPEPARS_OSGFILE_SET)) {
        vType->setAttribute(SUMO_ATTR_OSGFILE, toString(newVTypeParameter.osgFile), undoList);
    }
    if (newVTypeParameter.knowsParameter(toString(SUMO_ATTR_CARRIAGE_LENGTH))) {
        vType->setAttribute(SUMO_ATTR_CARRIAGE_LENGTH, newVTypeParameter.getParameter(toString(SUMO_ATTR_CARRIAGE_LENGTH), ""), undoList);
    }
    if (newVTypeParameter.knowsParameter(toString(SUMO_ATTR_LOCOMOTIVE_LENGTH))) {
        vType->setAttribute(SUMO_ATTR_LOCOMOTIVE_LENGTH, newVTypeParameter.getParameter(toString(SUMO_ATTR_LOCOMOTIVE_LENGTH), ""), undoList);
    }
    if (newVTypeParameter.knowsParameter(toString(SUMO_ATTR_CARRIAGE_GAP))) {
        vType->setAttribute(SUMO_ATTR_CARRIAGE_GAP, newVTypeParameter.getParameter(toString(SUMO_ATTR_CARRIAGE_GAP), ""), undoList);
    }
    // parse parameters
    std::string parametersStr;
    // Generate an string using the following structure: "key1=value1|key2=value2|...
    for (const auto& parameter : newVTypeParameter.getParametersMap()) {
        parametersStr += parameter.first + "=" + parameter.second + "|";
    }
    // remove the last "|"
    if (!parametersStr.empty()) {
        parametersStr.pop_back();
    }
    if (parametersStr != vType->getAttribute(GNE_ATTR_PARAMETERS)) {
        vType->setAttribute(GNE_ATTR_PARAMETERS, parametersStr, undoList);
    }
    if (parametersStr != vType->getAttribute(GNE_ATTR_VTYPE_DISTRIBUTION)) {
        vType->setAttribute(GNE_ATTR_VTYPE_DISTRIBUTION, parametersStr, undoList);
    }
    // close undo list
    undoList->end();
}

// ===========================================================================
// protected
// ===========================================================================

void
GNEVType::editVTypeDistribution(const std::string& vTypeDistributionID, GNEUndoList* undoList) {
    if (vTypeDistributionID.empty()) {
        // first check if previosly this vType has a vTypeDistribution
        if (getParentDemandElements().size() > 0) {
            // get parent vTypeDistribution
            GNEDemandElement* vTypeDistribution = getParentDemandElements().front();
            // check if this is the last vType of the vTypeDistribution
            if (vTypeDistribution->getChildDemandElements().size() == 1) {
                // ask if remove vTypeDistribution
                if (askRemoveVTypeDistribution(vTypeDistribution->getID())) {
                    undoList->begin(GUIIcon::VTYPEDISTRIBUTION, "remove " + toString(SUMO_TAG_VTYPE_DISTRIBUTION));
                    // clear attribute
                    undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, ""));
                    // remove vType Distribution
                    undoList->add(new GNEChange_DemandElement(vTypeDistribution, false), true);
                    undoList->end();
                }
            } else {
                // clear attribute
                undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, ""));
            }
        }
    } else {
        // check if vTypeDistribution exist
        const auto vTypeDistribution = myNet->getAttributeCarriers()->retrieveDemandElement(SUMO_TAG_VTYPE_DISTRIBUTION, vTypeDistributionID, false);
        // get current vTypeDistribution parent
        const auto vTypeDistributionParent = (getParentDemandElements().size() > 0) ? getParentDemandElements().front() : nullptr;
        if (vTypeDistribution) {
            // add in vTypeDistribution
            undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, vTypeDistributionID));
        } else if (vTypeDistributionParent && (vTypeDistributionParent->getChildDemandElements().size() == 1)) {
            // ask if remove vTypeDistribution
            if (askRemoveVTypeDistribution(vTypeDistributionParent->getID())) {
                undoList->begin(GUIIcon::VTYPEDISTRIBUTION, "add/remove " + toString(SUMO_TAG_VTYPE_DISTRIBUTION));
                // clear attribute
                undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, ""));
                // remove old vTypeDistribution
                undoList->add(new GNEChange_DemandElement(vTypeDistributionParent, false), true);
                // create newTypeDistribution
                undoList->add(new GNEChange_DemandElement(new GNEVTypeDistribution(myNet, vTypeDistributionID), true), true);
                // set new vTypeDistribution
                undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, vTypeDistributionID));
                undoList->end();
            }
        } else {
            undoList->begin(GUIIcon::VTYPEDISTRIBUTION, "add " + toString(SUMO_TAG_VTYPE_DISTRIBUTION));
            // create newTypeDistribution
            undoList->add(new GNEChange_DemandElement(new GNEVTypeDistribution(myNet, vTypeDistributionID), true), true);
            // set new vTypeDistribution
            undoList->changeAttribute(new GNEChange_Attribute(this, GNE_ATTR_VTYPE_DISTRIBUTION, vTypeDistributionID));
            undoList->end();
        }
    }
}


bool
GNEVType::askAddVTypeDistribution(const std::string& vTypeDistribution) const {
    // show warning in gui testing debug mode
    WRITE_DEBUG("Opening FXMessageBox 'add vTypeDistribution'");
    // Ask confirmation to user
    const FXuint answer = FXMessageBox::question(myNet->getViewNet()->getApp(), MBOX_YES_NO,
                          ("Add " + toString(SUMO_TAG_VTYPE_DISTRIBUTION) + "s").c_str(), "%s",
                          (toString(SUMO_TAG_VTYPE_DISTRIBUTION) + " '" + vTypeDistribution + "' doesn't exist. Create?").c_str());
    if (answer == 1) { // 1:yes, 2:no, 4:esc
        WRITE_DEBUG("Closed FXMessageBox 'add vTypeDistribution' with 'yes'");
        return true;
    } else {
        // write warning if netedit is running in testing mode
        if (answer == 2) {
            WRITE_DEBUG("Closed FXMessageBox 'add vTypeDistribution' with 'No'");
            return false;
        } else {
            WRITE_DEBUG("Closed FXMessageBox 'add vTypeDistribution' with 'ESC'");
            return false;
        }
    }
}


bool
GNEVType::askRemoveVTypeDistribution(const std::string& vTypeDistribution) const {
    // show warning in gui testing debug mode
    WRITE_DEBUG("Opening FXMessageBox 'remove vTypeDistribution'");
    // Ask confirmation to user
    const FXuint answer = FXMessageBox::question(myNet->getViewNet()->getApp(), MBOX_YES_NO,
                          ("Remove " + toString(SUMO_TAG_VTYPE_DISTRIBUTION)).c_str(), "%s",
                          ("Changing attribute will remove " + toString(SUMO_TAG_VTYPE_DISTRIBUTION) + " '" + vTypeDistribution + "'. Continue?").c_str());
    if (answer == 1) { // 1:yes, 2:no, 4:esc
        WRITE_DEBUG("Closed FXMessageBox 'remove vTypeDistribution' with 'yes'");
        return true;
    } else {
        // write warning if netedit is running in testing mode
        if (answer == 2) {
            WRITE_DEBUG("Closed FXMessageBox 'remove vTypeDistribution' with 'No'");
            return false;
        } else {
            WRITE_DEBUG("Closed FXMessageBox 'remove vTypeDistribution' with 'ESC'");
            return false;
        }
    }
}

// ===========================================================================
// private
// ===========================================================================

void
GNEVType::setAttribute(SumoXMLAttr key, const std::string& value) {
    // obtain default values depending of vehicle class
    VClassDefaultValues defaultValues(vehicleClass);
    switch (key) {
        case SUMO_ATTR_ID:
            // update microsimID
            setMicrosimID(value);
            // manually change VType parameters ID
            id = value;
            break;
        case GNE_ATTR_VTYPE_DISTRIBUTION:
            setVTypeDistributionParent(value);
            break;
        // CFM Attributes
        case SUMO_ATTR_ACCEL:
        case SUMO_ATTR_DECEL:
        case SUMO_ATTR_APPARENTDECEL:
        case SUMO_ATTR_EMERGENCYDECEL:
        case SUMO_ATTR_SIGMA:
        case SUMO_ATTR_TAU:
        case SUMO_ATTR_COLLISION_MINGAP_FACTOR:
        case SUMO_ATTR_TMP1:
        case SUMO_ATTR_TMP2:
        case SUMO_ATTR_TMP3:
        case SUMO_ATTR_TMP4:
        case SUMO_ATTR_TMP5:
        case SUMO_ATTR_CF_EIDM_USEVEHDYNAMICS:
        case SUMO_ATTR_CF_EIDM_MAX_VEH_PREVIEW:
        case SUMO_ATTR_CF_EIDM_T_LOOK_AHEAD:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_DRIVE:
        case SUMO_ATTR_CF_EIDM_T_REACTION:
        case SUMO_ATTR_CF_EIDM_T_PERSISTENCE_ESTIMATE:
        case SUMO_ATTR_CF_EIDM_C_COOLNESS:
        case SUMO_ATTR_CF_EIDM_SIG_LEADER:
        case SUMO_ATTR_CF_EIDM_SIG_GAP:
        case SUMO_ATTR_CF_EIDM_SIG_ERROR:
        case SUMO_ATTR_CF_EIDM_JERK_MAX:
        case SUMO_ATTR_CF_EIDM_EPSILON_ACC:
        case SUMO_ATTR_CF_EIDM_T_ACC_MAX:
        case SUMO_ATTR_CF_EIDM_M_FLATNESS:
        case SUMO_ATTR_CF_EIDM_M_BEGIN:
        case SUMO_ATTR_CF_PWAGNER2009_TAULAST:
        case SUMO_ATTR_CF_PWAGNER2009_APPROB:
        case SUMO_ATTR_CF_IDMM_ADAPT_FACTOR:
        case SUMO_ATTR_CF_IDMM_ADAPT_TIME:
        case SUMO_ATTR_CF_WIEDEMANN_SECURITY:
        case SUMO_ATTR_CF_WIEDEMANN_ESTIMATION:
        case SUMO_ATTR_TRAIN_TYPE:
        case SUMO_ATTR_K:
        case SUMO_ATTR_CF_KERNER_PHI:
        case SUMO_ATTR_CF_IDM_DELTA:
        case SUMO_ATTR_CF_IDM_STEPPING:
            // empty values means that value isn't set
            if (value.empty()) {
                const auto it = cfParameter.find(key);
                if (it != cfParameter.end()) {
                    cfParameter.erase(it);
                }
            } else {
                cfParameter[key] = value;
            }
            break;
        // JM Attributes
        case SUMO_ATTR_JM_CROSSING_GAP:
        case SUMO_ATTR_JM_IGNORE_KEEPCLEAR_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_YELLOW_TIME:
        case SUMO_ATTR_JM_DRIVE_AFTER_RED_TIME:
        case SUMO_ATTR_JM_DRIVE_RED_SPEED:
        case SUMO_ATTR_JM_IGNORE_FOE_PROB:
        case SUMO_ATTR_JM_IGNORE_FOE_SPEED:
        case SUMO_ATTR_JM_SIGMA_MINOR:
        case SUMO_ATTR_JM_TIMEGAP_MINOR:
            // empty values means that value isn't set
            if (value.empty()) {
                const auto it = jmParameter.find(key);
                if (it != jmParameter.end()) {
                    jmParameter.erase(it);
                }
            } else {
                jmParameter[key] = value;
            }
            break;
        case SUMO_ATTR_IMPATIENCE:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                impatience = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_IMPATIENCE_SET;
            } else {
                // set default value
                impatience = parse<double>(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_IMPATIENCE_SET;
            }
            break;
        // LCM Attributes
        case SUMO_ATTR_LCA_STRATEGIC_PARAM:
        case SUMO_ATTR_LCA_COOPERATIVE_PARAM:
        case SUMO_ATTR_LCA_SPEEDGAIN_PARAM:
        case SUMO_ATTR_LCA_KEEPRIGHT_PARAM:
        case SUMO_ATTR_LCA_SUBLANE_PARAM:
        case SUMO_ATTR_LCA_OPPOSITE_PARAM:
        case SUMO_ATTR_LCA_PUSHY:
        case SUMO_ATTR_LCA_PUSHYGAP:
        case SUMO_ATTR_LCA_ASSERTIVE:
        case SUMO_ATTR_LCA_IMPATIENCE:
        case SUMO_ATTR_LCA_ACCEL_LAT:
        case SUMO_ATTR_LCA_LOOKAHEADLEFT:
        case SUMO_ATTR_LCA_SPEEDGAINRIGHT:
        case SUMO_ATTR_LCA_MAXSPEEDLATSTANDING:
        case SUMO_ATTR_LCA_MAXSPEEDLATFACTOR:

            /* case SUMO_ATTR_LCA_EXPERIMENTAL1: */
            // empty values means that value isn't set
            if (value.empty()) {
                const auto it = lcParameter.find(key);
                if (it != lcParameter.end()) {
                    lcParameter.erase(it);
                }
            } else {
                lcParameter[key] = value;
            }
            break;
        case SUMO_ATTR_LCA_TURN_ALIGNMENT_DISTANCE:
        case SUMO_ATTR_LCA_TIME_TO_IMPATIENCE:
        case SUMO_ATTR_LCA_OVERTAKE_RIGHT:
            // empty or null values means that value isn't set
            if (value.empty() || (canParse<double>(value) && (parse<double>(value) == 0))) {
                const auto it = lcParameter.find(key);
                if (it != lcParameter.end()) {
                    lcParameter.erase(it);
                }
            } else {
                lcParameter[key] = value;
            }
            break;
        //
        case SUMO_ATTR_LENGTH:
            if (!value.empty() && (value != toString(defaultValues.length))) {
                length = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_LENGTH_SET;
            } else {
                // set default value
                length = defaultValues.length;
                // unset parameter
                parametersSet &= ~VTYPEPARS_LENGTH_SET;
            }
            break;
        case SUMO_ATTR_MINGAP:
            if (!value.empty() && (value != toString(defaultValues.minGap))) {
                minGap = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_MINGAP_SET;
            } else {
                // set default value
                minGap = defaultValues.minGap;
                // unset parameter
                parametersSet &= ~VTYPEPARS_MINGAP_SET;
            }
            break;
        case SUMO_ATTR_MAXSPEED:
            if (!value.empty() && (value != toString(defaultValues.maxSpeed))) {
                maxSpeed = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_MAXSPEED_SET;
            } else {
                // set default value
                maxSpeed = defaultValues.maxSpeed;
                // unset parameter
                parametersSet &= ~VTYPEPARS_MAXSPEED_SET;
            }
            break;
        case SUMO_ATTR_SPEEDFACTOR:
            if (!value.empty() && (value != toString(defaultValues.speedFactor))) {
                speedFactor.parse(value, false);
                // mark parameter as set
                parametersSet |= VTYPEPARS_SPEEDFACTOR_SET;
            } else {
                // set default value
                speedFactor.parse(toString(defaultValues.speedFactor), false);
                // unset parameter
                parametersSet &= ~VTYPEPARS_SPEEDFACTOR_SET;
            }
            break;
        case SUMO_ATTR_COLOR:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                color = parse<RGBColor>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_COLOR_SET;
            } else {
                // unset parameter
                parametersSet &= ~VTYPEPARS_COLOR_SET;
            }
            break;
        case SUMO_ATTR_VCLASS:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                vehicleClass = getVehicleClassID(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_VEHICLECLASS_SET;
            } else {
                // set default value
                vehicleClass = getVehicleClassID(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_VEHICLECLASS_SET;
            }
            // update default values
            updateDefaultVClassAttributes(defaultValues);
            break;
        case SUMO_ATTR_EMISSIONCLASS:
            if (!value.empty() && (value != toString(defaultValues.emissionClass))) {
                emissionClass = PollutantsInterface::getClassByName(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_EMISSIONCLASS_SET;
            } else {
                // set default value
                emissionClass = defaultValues.emissionClass;
                // unset parameter
                parametersSet &= ~VTYPEPARS_EMISSIONCLASS_SET;
            }
            break;
        case SUMO_ATTR_GUISHAPE:
            if (!value.empty() && (value != SumoVehicleShapeStrings.getString(defaultValues.shape))) {
                shape = getVehicleShapeID(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_SHAPE_SET;
            } else {
                // set default value
                shape = defaultValues.shape;
                // unset parameter
                parametersSet &= ~VTYPEPARS_SHAPE_SET;
            }
            break;
        case SUMO_ATTR_WIDTH:
            if (!value.empty() && (value != toString(defaultValues.width))) {
                width = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_WIDTH_SET;
            } else {
                // set default value
                width = defaultValues.width;
                // unset parameter
                parametersSet &= ~VTYPEPARS_WIDTH_SET;
            }
            break;
        case SUMO_ATTR_HEIGHT:
            if (!value.empty() && (value != toString(defaultValues.height))) {
                height = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_HEIGHT_SET;
            } else {
                // set default value
                height = defaultValues.height;
                // unset parameter
                parametersSet &= ~VTYPEPARS_HEIGHT_SET;
            }
            break;
        case SUMO_ATTR_IMGFILE:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                imgFile = value;
                // mark parameter as set
                parametersSet |= VTYPEPARS_IMGFILE_SET;
            } else {
                // set default value
                imgFile = myTagProperty.getDefaultValue(key);
                // unset parameter
                parametersSet &= ~VTYPEPARS_IMGFILE_SET;
            }
            break;
        case SUMO_ATTR_LANE_CHANGE_MODEL:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                lcModel = SUMOXMLDefinitions::LaneChangeModels.get(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_LANE_CHANGE_MODEL_SET;
            } else {
                // set default value
                lcModel = SUMOXMLDefinitions::LaneChangeModels.get(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_LANE_CHANGE_MODEL_SET;
            }
            break;
        case SUMO_ATTR_CAR_FOLLOW_MODEL:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                cfModel = SUMOXMLDefinitions::CarFollowModels.get(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_CAR_FOLLOW_MODEL;
            } else {
                // set default value
                cfModel = SUMOXMLDefinitions::CarFollowModels.get(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_CAR_FOLLOW_MODEL;
            }
            break;
        case SUMO_ATTR_PERSON_CAPACITY:
            if (!value.empty() && (value != toString(defaultValues.personCapacity))) {
                personCapacity = parse<int>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_PERSON_CAPACITY;
            } else {
                // set default value
                personCapacity = defaultValues.personCapacity;
                // unset parameter
                parametersSet &= ~VTYPEPARS_PERSON_CAPACITY;
            }
            break;
        case SUMO_ATTR_CONTAINER_CAPACITY:
            if (!value.empty() && (value != toString(defaultValues.containerCapacity))) {
                containerCapacity = parse<int>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_CONTAINER_CAPACITY;
            } else {
                // set default value
                containerCapacity = defaultValues.containerCapacity;
                // unset parameter
                parametersSet &= ~VTYPEPARS_CONTAINER_CAPACITY;
            }
            break;
        case SUMO_ATTR_BOARDING_DURATION:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                boardingDuration = string2time(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_BOARDING_DURATION;
            } else {
                // set default value
                boardingDuration = string2time(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_BOARDING_DURATION;
            }
            break;
        case SUMO_ATTR_LOADING_DURATION:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                loadingDuration = string2time(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_LOADING_DURATION;
            } else {
                // set default value
                loadingDuration = string2time(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_LOADING_DURATION;
            }
            break;
        case SUMO_ATTR_LATALIGNMENT:
            if (!value.empty() && (value != toString(defaultValues.latAlignmentProcedure))) {
                parseLatAlignment(value, latAlignmentOffset, latAlignmentProcedure);
                // mark parameter as set
                parametersSet |= VTYPEPARS_LATALIGNMENT_SET;
            } else {
                // set default value
                parseLatAlignment(toString(defaultValues.latAlignmentProcedure), latAlignmentOffset, latAlignmentProcedure);
                // unset parameter
                parametersSet &= ~VTYPEPARS_LATALIGNMENT_SET;
            }
            break;
        case SUMO_ATTR_MINGAP_LAT:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                minGapLat = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_MINGAP_LAT_SET;
            } else {
                // set default value
                minGapLat = parse<double>(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_MINGAP_LAT_SET;
            }
            break;
        case SUMO_ATTR_MAXSPEED_LAT:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                maxSpeedLat = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_MAXSPEED_LAT_SET;
            } else {
                // set default value
                maxSpeedLat = parse<double>(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_MAXSPEED_LAT_SET;
            }
            break;
        case SUMO_ATTR_ACTIONSTEPLENGTH:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                actionStepLength = string2time(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_ACTIONSTEPLENGTH_SET;
            } else {
                // set default value
                actionStepLength = string2time(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_ACTIONSTEPLENGTH_SET;
            }
            break;
        case SUMO_ATTR_PROB:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                defaultProbability = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_PROBABILITY_SET;
            } else {
                // set default value
                defaultProbability = parse<double>(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_PROBABILITY_SET;
            }
            break;
        case SUMO_ATTR_OSGFILE:
            if (!value.empty() && (value != defaultValues.osgFile)) {
                osgFile = value;
                // mark parameter as set
                parametersSet |= VTYPEPARS_OSGFILE_SET;
            } else {
                // set default value
                osgFile = defaultValues.osgFile;
                // unset parameter
                parametersSet &= ~VTYPEPARS_OSGFILE_SET;
            }
            break;
        case SUMO_ATTR_CARRIAGE_LENGTH:
            if (!value.empty() && (value != toString(defaultValues.carriageLength))) {
                carriageLength = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_CARRIAGE_LENGTH_SET;
                // set parameter in SUMOVTypeParameter (needed for writting in XML)
                SUMOVTypeParameter::setParameter(toString(key), value);
            } else {
                // set default value
                carriageLength = defaultValues.carriageLength;
                // unset parameter
                parametersSet &= ~VTYPEPARS_CARRIAGE_LENGTH_SET;
                // remove from params (needed for writting in XML)
                SUMOVTypeParameter::unsetParameter(toString(key));
            }
            break;
        case SUMO_ATTR_LOCOMOTIVE_LENGTH:
            if (!value.empty() && (value != toString(defaultValues.containerCapacity))) {
                locomotiveLength = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_LOCOMOTIVE_LENGTH_SET;
                // set parameter in SUMOVTypeParameter (needed for writting in XML)
                SUMOVTypeParameter::setParameter(toString(key), value);
            } else {
                // set default value
                locomotiveLength = defaultValues.locomotiveLength;
                // unset parameter
                parametersSet &= ~VTYPEPARS_LOCOMOTIVE_LENGTH_SET;
                // remove from params (needed for writting in XML)
                SUMOVTypeParameter::unsetParameter(toString(key));
            }
            break;
        case SUMO_ATTR_CARRIAGE_GAP:
            if (!value.empty() && (value != myTagProperty.getDefaultValue(key))) {
                carriageGap = parse<double>(value);
                // mark parameter as set
                parametersSet |= VTYPEPARS_CARRIAGE_GAP_SET;
                // set parameter in SUMOVTypeParameter (needed for writting in XML)
                SUMOVTypeParameter::setParameter(toString(key), value);
            } else {
                // set default value
                carriageGap = parse<double>(myTagProperty.getDefaultValue(key));
                // unset parameter
                parametersSet &= ~VTYPEPARS_CARRIAGE_GAP_SET;
                // remove from params (needed for writting in XML)
                SUMOVTypeParameter::unsetParameter(toString(key));
            }
            break;
        case GNE_ATTR_SELECTED:
            if (parse<bool>(value)) {
                selectAttributeCarrier();
            } else {
                unselectAttributeCarrier();
            }
            break;
        case GNE_ATTR_PARAMETERS:
            setParametersStr(value);
            break;
        case GNE_ATTR_DEFAULT_VTYPE_MODIFIED:
            myDefaultVehicleTypeModified = parse<bool>(value);
            break;
        default:
            throw InvalidArgument(getTagStr() + " doesn't have an attribute of type '" + toString(key) + "'");
    }
    // check if geometry must be marked as deprecated
    if (myTagProperty.hasAttribute(key) && (myTagProperty.getAttributeProperties(key).requireUpdateGeometry())) {
        updateGeometry();
    }
}


void
GNEVType::toogleAttribute(SumoXMLAttr /*key*/, const bool /*value*/) {
    // nothing to toogle
}


void
GNEVType::setMoveShape(const GNEMoveResult& /*moveResult*/) {
    // vehicleTypes cannot be moved
}


void
GNEVType::commitMoveShape(const GNEMoveResult& /*moveResult*/, GNEUndoList* /*undoList*/) {
    // vehicleTypes cannot be moved
}


void
GNEVType::updateDefaultVClassAttributes(const VClassDefaultValues& defaultValues) {
    if (!wasSet(VTYPEPARS_LENGTH_SET)) {
        length = defaultValues.length;
    }
    if (!wasSet(VTYPEPARS_MINGAP_SET)) {
        minGap = defaultValues.minGap;
    }
    if (!wasSet(VTYPEPARS_MAXSPEED_SET)) {
        maxSpeed = defaultValues.maxSpeed;
    }
    if (!wasSet(VTYPEPARS_WIDTH_SET)) {
        width = defaultValues.width;
    }
    if (!wasSet(VTYPEPARS_HEIGHT_SET)) {
        height = defaultValues.height;
    }
    if (!wasSet(VTYPEPARS_SHAPE_SET)) {
        shape = defaultValues.shape;
    }
    if (!wasSet(VTYPEPARS_EMISSIONCLASS_SET)) {
        emissionClass = defaultValues.emissionClass;
    }
    if (!wasSet(VTYPEPARS_SPEEDFACTOR_SET)) {
        speedFactor = defaultValues.speedFactor;
    }
    if (!wasSet(VTYPEPARS_PERSON_CAPACITY)) {
        personCapacity = defaultValues.personCapacity;
    }
    if (!wasSet(VTYPEPARS_CONTAINER_CAPACITY)) {
        containerCapacity = defaultValues.containerCapacity;
    }
    if (!wasSet(VTYPEPARS_OSGFILE_SET)) {
        osgFile = defaultValues.osgFile;
    }
    if (!wasSet(VTYPEPARS_CARRIAGE_LENGTH_SET)) {
        carriageLength = defaultValues.carriageLength;
    }
    if (!wasSet(VTYPEPARS_LOCOMOTIVE_LENGTH_SET)) {
        locomotiveLength = defaultValues.locomotiveLength;
    }
}

/****************************************************************************/
