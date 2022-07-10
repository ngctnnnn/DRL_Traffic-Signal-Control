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
/// @file    GUIGlObject.cpp
/// @author  Daniel Krajzewicz
/// @author  Jakob Erdmann
/// @author  Michael Behrisch
/// @author  Laura Bieker
/// @date    Sept 2002
///
// Base class for all objects that may be displayed within the openGL-gui
/****************************************************************************/
#include <config.h>

#include <string>
#include <stack>
#include <utils/common/ToString.h>
#include <utils/geom/GeoConvHelper.h>
#include <utils/gui/windows/GUISUMOAbstractView.h>
#include <utils/gui/globjects/GUIGLObjectPopupMenu.h>
#include <utils/gui/div/GUIParameterTableWindow.h>
#include <utils/foxtools/MFXMenuHeader.h>
#include <utils/gui/images/GUIIconSubSys.h>
#include <utils/gui/windows/GUIAppEnum.h>
#include <utils/gui/windows/GUIMainWindow.h>
#include <utils/gui/div/GUIGlobalSelection.h>
#include <utils/gui/div/GLHelper.h>
#include <utils/gui/div/GLObjectValuePassConnector.h>
#include <utils/gui/div/GUIDesigns.h>

#include "GUIGlObject.h"
#include "GUIGlObjectStorage.h"

// ===========================================================================
// static members
// ===========================================================================

StringBijection<GUIGlObjectType>::Entry GUIGlObject::GUIGlObjectTypeNamesInitializer[] = {
    {"network",                 GLO_NETWORK},
    //
    {"networkElement",          GLO_NETWORKELEMENT},
    {"edge",                    GLO_EDGE},
    {"lane",                    GLO_LANE},
    {"junction",                GLO_JUNCTION},
    {"connection",              GLO_CONNECTION},
    {"crossing",                GLO_CROSSING},
    {"walkingArea",             GLO_WALKINGAREA},
    {"tlLogic",                 GLO_TLLOGIC},
    {"type",                    GLO_TYPE},
    //
    {"parentChildLine",         GLO_PARENTCHILDLINE},
    //
    {"additional",              GLO_ADDITIONALELEMENT},
    {"busStop",                 GLO_BUS_STOP},
    {"access",                  GLO_ACCESS},
    {"taz",                     GLO_TAZ},
    {"containerStop",           GLO_CONTAINER_STOP},
    {"chargingStation",         GLO_CHARGING_STATION},
    {"overheadWireSegment",     GLO_OVERHEAD_WIRE_SEGMENT},
    {"parkingArea",             GLO_PARKING_AREA},
    {"parkingSpace",            GLO_PARKING_SPACE},
    {"e1Detector",              GLO_E1DETECTOR},
    {"e1DetectorME",            GLO_E1DETECTOR_ME},
    {"e1DetectorInstant",       GLO_E1DETECTOR_INSTANT},
    {"e2Detector",              GLO_E2DETECTOR},
    {"e3Detector",              GLO_E3DETECTOR},
    {"entryDetector",           GLO_DET_ENTRY},
    {"exitDetector",            GLO_DET_EXIT},
    {"rerouter",                GLO_REROUTER},
    {"rerouterInterval",        GLO_REROUTER_INTERVAL},
    {"closingreroute",          GLO_REROUTER_CLOSINGREROUTE},
    {"closingLaneReroute",      GLO_REROUTER_CLOSINGLANEREROUTE},
    {"parkingAreaReroute",      GLO_REROUTER_PARKINGAREAREROUTE},
    {"destProbReroute",         GLO_REROUTER_DESTPROBREROUTE},
    {"routeProbReroute",        GLO_REROUTER_ROUTEPROBREROUTE},
    {"rerouterEdge",            GLO_REROUTER_EDGE},
    {"variableSpeedSign",       GLO_VSS},
    {"variableSpeedSignStep",   GLO_VSS_STEP},
    {"calibrator",              GLO_CALIBRATOR},
    {"routeProbe",              GLO_ROUTEPROBE},
    {"vaporizer",               GLO_VAPORIZER},
    {"wire",                    GLO_WIRE},
    {"tractionsubstation",      GLO_TRACTIONSUBSTATION},
    //
    {"shape",                   GLO_SHAPE},
    {"polygon",                 GLO_POLYGON},
    {"poi",                     GLO_POI},
    //
    {"routeElement",            GLO_ROUTEELEMENT},
    {"vType",                   GLO_VTYPE},
    //
    {"route",                   GLO_ROUTE},
    //
    {"ride",                    GLO_RIDE},
    {"walk",                    GLO_WALK},
    {"personTrip",              GLO_PERSONTRIP},
    {"transport",               GLO_TRANSPORT},
    {"tranship",                GLO_TRANSHIP},
    //
    {"stop",                    GLO_STOP},
    {"stopPerson",              GLO_STOP_PERSON},
    {"stopContainer",           GLO_STOP_CONTAINER},
    //
    {"vehicle",                 GLO_VEHICLE},
    {"trip",                    GLO_TRIP},
    {"flow",                    GLO_FLOW},
    {"routeFlow",               GLO_ROUTEFLOW},
    //
    {"container",               GLO_CONTAINER},
    {"containerFlow",           GLO_CONTAINERFLOW},
    //
    {"person",                  GLO_PERSON},
    {"personFlow",              GLO_PERSONFLOW},
    //
    {"edgeData",                GLO_EDGEDATA},
    {"edgeRelData",             GLO_EDGERELDATA},
    {"TAZRelData",              GLO_TAZRELDATA},
    //
    {"lockIcon",                GLO_LOCKICON},
    {"textName",                GLO_TEXTNAME},
    {"frontElement",            GLO_DOTTEDCONTOUR_FRONT},
    {"dottedContour",           GLO_DOTTEDCONTOUR_INSPECTED},
    {"temporalShape",           GLO_TEMPORALSHAPE},
    {"rectangleSelection",      GLO_RECTANGLESELECTION},
    {"testElement",             GLO_TESTELEMENT},
    //
    {"undefined",               GLO_MAX}
};


StringBijection<GUIGlObjectType> GUIGlObject::TypeNames(GUIGlObjectTypeNamesInitializer, GLO_MAX);
const GUIGlID GUIGlObject::INVALID_ID = 0;

// ===========================================================================
// method definitionsas
// ===========================================================================

GUIGlObject::GUIGlObject(GUIGlObjectType type, const std::string& microsimID) :
    myGlID(GUIGlObjectStorage::gIDStorage.registerObject(this)),
    myGLObjectType(type),
    myMicrosimID(microsimID),
    myAmBlocked(false) {
    // make sure that reserved GLO_ADDITIONALELEMENT isn't used
    assert(myGLObjectType != GLO_ADDITIONALELEMENT);
    myFullName = createFullName();
    GUIGlObjectStorage::gIDStorage.changeName(this, myFullName);
}


GUIGlObject::~GUIGlObject() {
    // remove all paramWindow related with this object
    for (const auto& paramWindow : myParamWindows) {
        paramWindow->removeObject(this);
    }
    // remove object from GLObjectValuePassConnector and GUIGlObjectStorage
    GLObjectValuePassConnector<double>::removeObject(*this);
    GUIGlObjectStorage::gIDStorage.remove(getGlID());
}


std::string
GUIGlObject::getParentName() const {
    return StringUtils::emptyString;
}


GUIParameterTableWindow*
GUIGlObject::getTypeParameterWindow(GUIMainWindow& app, GUISUMOAbstractView& parent) {
    UNUSED_PARAMETER(&app);
    UNUSED_PARAMETER(&parent);
    return nullptr;
}


const std::string
GUIGlObject::getOptionalName() const {
    return "";
}


void
GUIGlObject::setMicrosimID(const std::string& newID) {
    myMicrosimID = newID;
    GUIGlObjectStorage::gIDStorage.changeName(this, createFullName());
    myFullName = createFullName();
}


void
GUIGlObject::drawGLAdditional(GUISUMOAbstractView* const parent, const GUIVisualizationSettings& s) const {
    UNUSED_PARAMETER(&s);
    UNUSED_PARAMETER(parent);
}

#ifdef HAVE_OSG

osg::Node*
GUIGlObject::getNode() const {
    return myOSGNode;
}


void
GUIGlObject::setNode(osg::Node* node) {
    myOSGNode = node;
}

#endif

void
GUIGlObject::buildPopupHeader(GUIGLObjectPopupMenu* ret, GUIMainWindow& app, bool addSeparator) {
    new MFXMenuHeader(ret, app.getBoldFont(), getFullName().c_str(), nullptr, nullptr, 0);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildCenterPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    GUIDesigns::buildFXMenuCommand(ret, "Center", GUIIconSubSys::getIcon(GUIIcon::RECENTERVIEW), ret, MID_CENTER);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildNameCopyPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    GUIDesigns::buildFXMenuCommand(ret, "Copy name to clipboard", nullptr, ret, MID_COPY_NAME);
    GUIDesigns::buildFXMenuCommand(ret, "Copy typed name to clipboard", nullptr, ret, MID_COPY_TYPED_NAME);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildSelectionPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    if (gSelected.isSelected(getType(), getGlID())) {
        GUIDesigns::buildFXMenuCommand(ret, "Remove From Selected", GUIIconSubSys::getIcon(GUIIcon::FLAG_MINUS), ret, MID_REMOVESELECT);
    } else {
        GUIDesigns::buildFXMenuCommand(ret, "Add To Selected", GUIIconSubSys::getIcon(GUIIcon::FLAG_PLUS), ret, MID_ADDSELECT);
    }
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildShowParamsPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    GUIDesigns::buildFXMenuCommand(ret, "Show Parameter", GUIIconSubSys::getIcon(GUIIcon::APP_TABLE), ret, MID_SHOWPARS);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildShowTypeParamsPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    GUIDesigns::buildFXMenuCommand(ret, "Show Type Parameter", GUIIconSubSys::getIcon(GUIIcon::APP_TABLE), ret, MID_SHOWTYPEPARS);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildPositionCopyEntry(GUIGLObjectPopupMenu* ret, const GUIMainWindow& app) const {
    GUIDesigns::buildFXMenuCommand(ret, "Copy cursor position to clipboard", nullptr, ret, MID_COPY_CURSOR_POSITION);
    if (GeoConvHelper::getFinal().usingGeoProjection()) {
        GUIDesigns::buildFXMenuCommand(ret, "Copy cursor geo-position to clipboard", nullptr, ret, MID_COPY_CURSOR_GEOPOSITION);
        for (const auto& mapper : app.getOnlineMaps()) {
            GUIDesigns::buildFXMenuCommand(ret, "Show cursor geo-position in " + mapper.first, nullptr, ret, MID_SHOW_GEOPOSITION_ONLINE);
        }
    }
}


void
GUIGlObject::buildShowManipulatorPopupEntry(GUIGLObjectPopupMenu* ret, bool addSeparator) {
    GUIDesigns::buildFXMenuCommand(ret, "Open Manipulator...", GUIIconSubSys::getIcon(GUIIcon::MANIP), ret, MID_MANIP);
    if (addSeparator) {
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::addParameterTable(GUIParameterTableWindow* t) {
    myParamWindows.insert(t);
}


void
GUIGlObject::removeParameterTable(GUIParameterTableWindow* t) {
    std::set<GUIParameterTableWindow*>::iterator i = myParamWindows.find(t);
    if (i != myParamWindows.end()) {
        myParamWindows.erase(i);
    }
}


void
GUIGlObject::buildShapePopupOptions(GUIMainWindow& app, GUIGLObjectPopupMenu* ret, const std::string& type) {
    assert(ret);
    // build header (<tag>:<ID>
    buildPopupHeader(ret, app, false);
    // build center
    buildCenterPopupEntry(ret);
    // build copy name
    buildNameCopyPopupEntry(ret);
    // build select/unselect
    buildSelectionPopupEntry(ret);
    // build show parameters
    buildShowParamsPopupEntry(ret, false);
    // build copy cursor position to clipboard
    buildPositionCopyEntry(ret, app);
    // only show type if isn't empty
    if (type != "") {
        GUIDesigns::buildFXMenuCommand(ret, ("type: " + type + "").c_str(), nullptr, nullptr, 0);
        new FXMenuSeparator(ret);
    }
}


void
GUIGlObject::buildAdditionalsPopupOptions(GUIMainWindow& app, GUIGLObjectPopupMenu* ret, const std::string& type) {
    assert(ret);
    // build header (<tag>:<ID>
    buildPopupHeader(ret, app, false);
    // build center
    buildCenterPopupEntry(ret);
    // build copy name
    buildNameCopyPopupEntry(ret);
    // build select/unselect
    buildSelectionPopupEntry(ret);
    // build show parameters
    buildShowParamsPopupEntry(ret, false);
    // build copy cursor position to clipboard
    buildPositionCopyEntry(ret, app);
    // only show type if isn't empty
    if (type != "") {
        GUIDesigns::buildFXMenuCommand(ret, ("type: " + type + "").c_str(), nullptr, nullptr, 0);
        new FXMenuSeparator(ret);
    }
}


std::string
GUIGlObject::createFullName() const {
    return TypeNames.getString(myGLObjectType) + ":" + getMicrosimID();
}


void
GUIGlObject::drawName(const Position& pos, const double scale, const GUIVisualizationTextSettings& settings, const double angle, bool forceShow) const {
    if (settings.show(this) || forceShow) {
        GLHelper::drawTextSettings(settings, getMicrosimID(), pos, scale, angle);
    }
}


/****************************************************************************/
