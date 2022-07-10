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
/// @file    GNEDataHandler.cpp
/// @author  Pablo Alvarez Lopez
/// @date    Jan 2020
///
// Builds data objects for netedit
/****************************************************************************/

// ===========================================================================
// included modules
// ===========================================================================
#include <config.h>

#include <netedit/changes/GNEChange_DataSet.h>
#include <netedit/changes/GNEChange_DataInterval.h>
#include <netedit/changes/GNEChange_GenericData.h>
#include <netedit/elements/data/GNEEdgeRelData.h>
#include <netedit/elements/data/GNEEdgeData.h>
#include <netedit/elements/data/GNETAZRelData.h>
#include <netedit/elements/data/GNEDataInterval.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNENet.h>
#include <netedit/GNEUndoList.h>
#include <utils/gui/div/GUIDesigns.h>

#include "GNEDataHandler.h"


// ===========================================================================
// member method definitions
// ===========================================================================

GNEDataHandler::GNEDataHandler(GNENet* net, const std::string& file, const bool allowUndoRedo) :
    DataHandler(file),
    myNet(net),
    myAllowUndoRedo(allowUndoRedo) {
}


GNEDataHandler::~GNEDataHandler() {}


void
GNEDataHandler::buildDataSet(const std::string& dataSetID) {
    // first check if dataSet exist
    if (myNet->getAttributeCarriers()->retrieveDataSet(dataSetID, false) == nullptr) {
        GNEDataSet* dataSet = new GNEDataSet(myNet, dataSetID);
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::DATASET, "add " + toString(SUMO_TAG_DATASET));
            myNet->getViewNet()->getUndoList()->add(new GNEChange_DataSet(dataSet, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert dataSet without allowing undo/redo
            myNet->getAttributeCarriers()->insertDataSet(dataSet);
            dataSet->incRef("buildDataSet");
        }
    } else {
        writeErrorDuplicated(SUMO_TAG_DATASET, dataSetID);
    }
}


void
GNEDataHandler::buildDataInterval(const CommonXMLStructure::SumoBaseObject* /* sumoBaseObject */,
                                  const std::string& dataSetID, const double begin, const double end) {
    // get dataSet
    GNEDataSet* dataSet = myNet->getAttributeCarriers()->retrieveDataSet(dataSetID, false);
    // first check if dataSet exist
    if (dataSet == nullptr) {
        // create dataset AND data interval
        dataSet = new GNEDataSet(myNet, dataSetID);
        GNEDataInterval* dataInterval = new GNEDataInterval(dataSet, begin, end);
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::DATASET, "add " + toString(SUMO_TAG_DATASET) + " and " + toString(SUMO_TAG_DATAINTERVAL));
            myNet->getViewNet()->getUndoList()->add(new GNEChange_DataSet(dataSet, true), true);
            myNet->getViewNet()->getUndoList()->add(new GNEChange_DataInterval(dataInterval, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert dataInterval without allowing undo/redo
            myNet->getAttributeCarriers()->insertDataSet(dataSet);
            dataSet->addDataIntervalChild(dataInterval);
            dataInterval->incRef("buildDataInterval");
        }
    } else if (dataSet->retrieveInterval(begin, end) == nullptr) {
        GNEDataInterval* dataInterval = new GNEDataInterval(dataSet, begin, end);
        if (myAllowUndoRedo) {
            myNet->getViewNet()->getUndoList()->begin(GUIIcon::DATAINTERVAL, "add " + toString(SUMO_TAG_DATAINTERVAL));
            myNet->getViewNet()->getUndoList()->add(new GNEChange_DataInterval(dataInterval, true), true);
            myNet->getViewNet()->getUndoList()->end();
        } else {
            // insert dataInterval without allowing undo/redo
            dataSet->addDataIntervalChild(dataInterval);
            dataInterval->incRef("buildDataInterval");
        }
    }
}


void
GNEDataHandler::buildEdgeData(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& edgeID,
                              const Parameterised::Map& parameters) {
    // get dataSet
    GNEDataSet* dataSet = myNet->getAttributeCarriers()->retrieveDataSet(sumoBaseObject->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID), false);
    if (dataSet != nullptr) {
        // get interval
        GNEDataInterval* dataInterval = dataSet->retrieveInterval(
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_BEGIN),
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_END));
        if (dataInterval != nullptr) {
            // get data
            GNEEdge* edge = myNet->getAttributeCarriers()->retrieveEdge(edgeID, false);
            if (edge) {
                GNEGenericData* edgeData = new GNEEdgeData(dataInterval, edge, parameters);
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::EDGEDATA, "add " + toString(SUMO_TAG_MEANDATA_EDGE));
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_GenericData(edgeData, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    dataInterval->addGenericDataChild(edgeData);
                    edge->addChildElement(edgeData);
                    edgeData->incRef("buildEdgeData");
                }
            } else {
                writeErrorInvalidParent(SUMO_TAG_MEANDATA_EDGE, SUMO_TAG_EDGE);
            }
        } else {
            writeErrorInvalidParent(SUMO_TAG_MEANDATA_EDGE, SUMO_TAG_DATAINTERVAL);
        }
    } else {
        writeErrorInvalidParent(SUMO_TAG_MEANDATA_EDGE, SUMO_TAG_DATASET);
    }
}


void
GNEDataHandler::buildEdgeRelationData(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& fromEdgeID,
                                      const std::string& toEdgeID, const Parameterised::Map& parameters) {
    // get dataSet
    GNEDataSet* dataSet = myNet->getAttributeCarriers()->retrieveDataSet(sumoBaseObject->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID), false);
    if (dataSet != nullptr) {
        // get interval
        GNEDataInterval* dataInterval = dataSet->retrieveInterval(
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_BEGIN),
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_END));
        if (dataInterval != nullptr) {
            // get data
            GNEEdge* fromEdge = myNet->getAttributeCarriers()->retrieveEdge(fromEdgeID, false);
            GNEEdge* toEdge = myNet->getAttributeCarriers()->retrieveEdge(toEdgeID, false);
            if (fromEdge && toEdge) {
                GNEGenericData* edgeData = new GNEEdgeRelData(dataInterval, fromEdge, toEdge, parameters);
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::EDGERELDATA, "add " + toString(SUMO_TAG_MEANDATA_EDGE));
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_GenericData(edgeData, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    dataInterval->addGenericDataChild(edgeData);
                    fromEdge->addChildElement(edgeData);
                    toEdge->addChildElement(edgeData);
                    edgeData->incRef("buildEdgeData");
                }
            } else {
                writeErrorInvalidParent(SUMO_TAG_EDGEREL, SUMO_TAG_EDGE);
            }
        } else {
            writeErrorInvalidParent(SUMO_TAG_EDGEREL, SUMO_TAG_DATAINTERVAL);
        }
    } else {
        writeErrorInvalidParent(SUMO_TAG_EDGEREL, SUMO_TAG_DATASET);
    }
}


void
GNEDataHandler::buildTAZRelationData(const CommonXMLStructure::SumoBaseObject* sumoBaseObject, const std::string& fromTAZID,
                                     const std::string& toTAZID, const Parameterised::Map& parameters) {
    // get dataSet
    GNEDataSet* dataSet = myNet->getAttributeCarriers()->retrieveDataSet(sumoBaseObject->getParentSumoBaseObject()->getStringAttribute(SUMO_ATTR_ID), false);
    if (dataSet != nullptr) {
        // get interval
        GNEDataInterval* dataInterval = dataSet->retrieveInterval(
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_BEGIN),
                                            sumoBaseObject->getParentSumoBaseObject()->getDoubleAttribute(SUMO_ATTR_END));
        if (dataInterval != nullptr) {
            // get from TAZs
            GNEAdditional* fromTAZ = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TAZ, fromTAZID, false);
            GNEAdditional* toTAZ = myNet->getAttributeCarriers()->retrieveAdditional(SUMO_TAG_TAZ, toTAZID, false);
            if (fromTAZ == nullptr) {
                writeErrorInvalidParent(SUMO_TAG_TAZREL, SUMO_TAG_TAZ, fromTAZID);
            } else if (toTAZ == nullptr) {
                writeErrorInvalidParent(SUMO_TAG_TAZREL, SUMO_TAG_TAZ, toTAZID);
            } else if ((fromTAZ != toTAZ) && dataInterval->TAZRelExists(fromTAZ, toTAZ)) {
                WRITE_ERROR("There is already a " + toString(SUMO_TAG_TAZREL) + " defined between '" + toTAZID + "' and '" + toTAZID + "'.");
            } else if ((fromTAZ == toTAZ) && dataInterval->TAZRelExists(fromTAZ)) {
                WRITE_ERROR("There is already a " + toString(SUMO_TAG_TAZREL) + " defined in '" + toTAZID + "'.");
            } else if (fromTAZ == toTAZ) {
                GNEGenericData* edgeData = new GNETAZRelData(dataInterval, fromTAZ, parameters);
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZRELDATA, "add " + toString(SUMO_TAG_TAZREL));
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_GenericData(edgeData, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    dataInterval->addGenericDataChild(edgeData);
                    fromTAZ->addChildElement(edgeData);
                    edgeData->incRef("buildEdgeData");
                }
            } else {
                GNEGenericData* edgeData = new GNETAZRelData(dataInterval, fromTAZ, toTAZ, parameters);
                if (myAllowUndoRedo) {
                    myNet->getViewNet()->getUndoList()->begin(GUIIcon::TAZRELDATA, "add " + toString(SUMO_TAG_TAZREL));
                    myNet->getViewNet()->getUndoList()->add(new GNEChange_GenericData(edgeData, true), true);
                    myNet->getViewNet()->getUndoList()->end();
                } else {
                    dataInterval->addGenericDataChild(edgeData);
                    fromTAZ->addChildElement(edgeData);
                    toTAZ->addChildElement(edgeData);
                    edgeData->incRef("buildEdgeData");
                }
            }
        } else {
            writeErrorInvalidParent(SUMO_TAG_TAZREL, SUMO_TAG_DATAINTERVAL);
        }
    } else {
        writeErrorInvalidParent(SUMO_TAG_TAZREL, SUMO_TAG_DATASET);
    }
}


void
GNEDataHandler::writeErrorDuplicated(const SumoXMLTag tag, const std::string& id) const {
    WRITE_ERROR("Could not build " + toString(tag) + " with ID '" + id + "' in netedit; declared twice.");
}


void
GNEDataHandler::writeErrorInvalidParent(const SumoXMLTag tag, const SumoXMLTag parent) const {
    WRITE_ERROR("Could not build " + toString(tag) + " in netedit; " +  toString(parent) + " doesn't exist.");
}


void
GNEDataHandler::writeErrorInvalidParent(const SumoXMLTag tag, const SumoXMLTag parent, const std::string& ID) const {
    WRITE_ERROR("Could not build " + toString(tag) + " in netedit; " +  toString(parent) + " with ID '" + ID + "' doesn't exist.");
}

/****************************************************************************/
