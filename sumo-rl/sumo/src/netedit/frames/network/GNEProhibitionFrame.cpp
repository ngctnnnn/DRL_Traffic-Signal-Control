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
/// @file    GNEProhibitionFrame.cpp
/// @author  Mirko Barthauer (Technische Universitaet Braunschweig)
/// @date    May 2018
///
// The Widget for editing connection prohibits
/****************************************************************************/
#include <config.h>

#include <utils/gui/div/GUIDesigns.h>
#include <utils/gui/windows/GUIAppEnum.h>
#include <netedit/elements/network/GNELane.h>
#include <netedit/elements/network/GNEConnection.h>
#include <netedit/elements/network/GNEEdge.h>
#include <netedit/elements/network/GNEJunction.h>
#include <netedit/GNEViewNet.h>

#include "GNEProhibitionFrame.h"

// ===========================================================================
// FOX callback mapping
// ===========================================================================

FXDEFMAP(GNEProhibitionFrame) GNEProhibitionFrameMap[] = {
    FXMAPFUNC(SEL_COMMAND,  MID_CANCEL,     GNEProhibitionFrame::onCmdCancel),
    FXMAPFUNC(SEL_COMMAND,  MID_OK,         GNEProhibitionFrame::onCmdOK)
};

// Object implementation
FXIMPLEMENT(GNEProhibitionFrame, FXVerticalFrame, GNEProhibitionFrameMap, ARRAYNUMBER(GNEProhibitionFrameMap))

// ===========================================================================
// method definitions
// ===========================================================================

// ---------------------------------------------------------------------------
// GNEProhibitionFrame::RelativeToConnection - methods
// ---------------------------------------------------------------------------

GNEProhibitionFrame::RelativeToConnection::RelativeToConnection(GNEProhibitionFrame* prohibitionFrameParent) :
    FXGroupBoxModule(prohibitionFrameParent, "Relative to connection"),
    myProhibitionFrameParent(prohibitionFrameParent) {
    // Create label for current connection description and update it
    myConnDescriptionLabel = new FXLabel(getCollapsableFrame(), "", nullptr, GUIDesignLabelFrameInformation);
    // update description
    updateDescription();
}


GNEProhibitionFrame::RelativeToConnection::~RelativeToConnection() {}


void
GNEProhibitionFrame::RelativeToConnection::updateDescription() const {
    // update depending of myCurrentConn
    if (myProhibitionFrameParent->myCurrentConn == nullptr) {
        myConnDescriptionLabel->setText("No Connection selected\n");
    } else {
        myConnDescriptionLabel->setText(("from lane " + myProhibitionFrameParent->myCurrentConn->getLaneFrom()->getMicrosimID() +
                                         "\nto lane " + myProhibitionFrameParent->myCurrentConn->getLaneTo()->getMicrosimID()).c_str());
    }
}

// ---------------------------------------------------------------------------
// GNEProhibitionFrame::ProhibitionLegend - methods
// ---------------------------------------------------------------------------

GNEProhibitionFrame::Legend::Legend(GNEProhibitionFrame* prohibitionFrameParent) :
    FXGroupBoxModule(prohibitionFrameParent, "Information"),
    myUndefinedColor(RGBColor::GREY),
    myProhibitedColor(RGBColor(0, 179, 0)),
    myProhibitingColor(RGBColor::RED),
    myUnregulatedConflictColor(RGBColor::ORANGE),
    myMutualConflictColor(RGBColor::CYAN) {
    // Create labels for color legend
    FXLabel* legendLabel = new FXLabel(getCollapsableFrame(), "Selected", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setTextColor(MFXUtils::getFXColor(RGBColor::WHITE));
    legendLabel->setBackColor(MFXUtils::getFXColor(prohibitionFrameParent->myViewNet->getVisualisationSettings().colorSettings.selectedProhibitionColor));
    // label for conflicts
    legendLabel = new FXLabel(getCollapsableFrame(), "No conflict", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setBackColor(MFXUtils::getFXColor(myUndefinedColor));
    // label for yields
    legendLabel = new FXLabel(getCollapsableFrame(), "Yields", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setBackColor(MFXUtils::getFXColor(myProhibitedColor));
    // label for right of way
    legendLabel = new FXLabel(getCollapsableFrame(), "Has right of way", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setBackColor(MFXUtils::getFXColor(myProhibitingColor));
    // label for unregulated conflict
    legendLabel = new FXLabel(getCollapsableFrame(), "Unregulated conflict", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setBackColor(MFXUtils::getFXColor(myUnregulatedConflictColor));
    // label for mutual conflict
    legendLabel = new FXLabel(getCollapsableFrame(), "Mutual conflict", nullptr, GUIDesignLabelFrameInformation);
    legendLabel->setBackColor(MFXUtils::getFXColor(myMutualConflictColor));
}


GNEProhibitionFrame::Legend::~Legend() {}


const RGBColor&
GNEProhibitionFrame::Legend::getUndefinedColor() const {
    return myUndefinedColor;
}


const RGBColor&
GNEProhibitionFrame::Legend::getProhibitedColor() const {
    return myProhibitedColor;
}


const RGBColor&
GNEProhibitionFrame::Legend::getProhibitingColor() const {
    return myProhibitingColor;
}


const RGBColor&
GNEProhibitionFrame::Legend::getUnregulatedConflictColor() const {
    return myUnregulatedConflictColor;
}


const RGBColor&
GNEProhibitionFrame::Legend::getMutualConflictColor() const {
    return myMutualConflictColor;
}

// ---------------------------------------------------------------------------
// GNEProhibitionFrame::Modifications - methods
// ---------------------------------------------------------------------------

GNEProhibitionFrame::Modifications::Modifications(GNEProhibitionFrame* prohibitionFrameParent) :
    FXGroupBoxModule(prohibitionFrameParent, "Modifications") {

    // Create "OK" button
    mySaveButton = new FXButton(getCollapsableFrame(), "OK\t\tSave prohibition modifications (Enter)",
                                GUIIconSubSys::getIcon(GUIIcon::ACCEPT), prohibitionFrameParent, MID_OK, GUIDesignButton);

    // Create "Cancel" button
    myCancelButton = new FXButton(getCollapsableFrame(), "Cancel\t\tDiscard prohibition modifications (Esc)",
                                  GUIIconSubSys::getIcon(GUIIcon::CANCEL), prohibitionFrameParent, MID_CANCEL, GUIDesignButton);

    // Currently mySaveButton is disabled
    mySaveButton->disable();
    mySaveButton->hide();
}


GNEProhibitionFrame::Modifications::~Modifications() {}

// ---------------------------------------------------------------------------
// GNEProhibitionFrame - methods
// ---------------------------------------------------------------------------

GNEProhibitionFrame::GNEProhibitionFrame(FXHorizontalFrame* horizontalFrameParent, GNEViewNet* viewNet) :
    GNEFrame(horizontalFrameParent, viewNet, "Prohibits"),
    myCurrentConn(nullptr) {
    // set frame header label
    getFrameHeaderLabel()->setText("Prohibitions");

    // create RelativeToConnection
    myRelativeToConnection = new RelativeToConnection(this);

    // create legend
    myLegend = new Legend(this);

    // create Modifications
    myModifications = new Modifications(this);
}


GNEProhibitionFrame::~GNEProhibitionFrame() {}


void
GNEProhibitionFrame::handleProhibitionClick(const GNEViewNetHelper::ObjectsUnderCursor& objectsUnderCursor) {
    // build prohibition
    buildProhibition(objectsUnderCursor.getConnectionFront(), myViewNet->getMouseButtonKeyPressed().shiftKeyPressed(), myViewNet->getMouseButtonKeyPressed().controlKeyPressed(), true);
}


void
GNEProhibitionFrame::show() {
    GNEFrame::show();
}


void
GNEProhibitionFrame::hide() {
    GNEFrame::hide();
}


long
GNEProhibitionFrame::onCmdCancel(FXObject*, FXSelector, void*) {
    if (myCurrentConn != nullptr) {
        for (auto conn : myConcernedConns) {
            conn->setSpecialColor(nullptr);
        }
        myCurrentConn->setSpecialColor(nullptr);
        myCurrentConn = nullptr;
        myConcernedConns.clear();
        myRelativeToConnection->updateDescription();
        myViewNet->updateViewNet();
    }
    return 1;
}


long
GNEProhibitionFrame::onCmdOK(FXObject*, FXSelector, void*) {
    return 1;
}

// ---------------------------------------------------------------------------
// GNEProhibitionFrame - private methods
// ---------------------------------------------------------------------------

void
GNEProhibitionFrame::buildProhibition(GNEConnection* conn, bool /* mayDefinitelyPass */, bool /* allowConflict */, bool /* toggle */) {
    if (myCurrentConn == nullptr) {
        myCurrentConn = conn;
        myCurrentConn->setSpecialColor(&myViewNet->getVisualisationSettings().colorSettings.selectedProhibitionColor);

        // determine prohibition status of all other connections with respect to the selected one
        GNEJunction* junction = myCurrentConn->getEdgeFrom()->getToJunction();
        std::vector<GNEConnection*> allConns = junction->getGNEConnections();
        NBNode* node = junction->getNBNode();
        NBEdge* currentConnFrom = myCurrentConn->getEdgeFrom()->getNBEdge();

        const int currentLinkIndex = node->getConnectionIndex(currentConnFrom, myCurrentConn->getNBEdgeConnection());
        std::string currentFoesString = node->getFoes(currentLinkIndex);
        std::string currentResponseString = node->getResponse(currentLinkIndex);
        std::reverse(currentFoesString.begin(), currentFoesString.end());
        std::reverse(currentResponseString.begin(), currentResponseString.end());
        // iterate over all connections
        for (const auto& i : allConns) {
            if (i != myCurrentConn) {
                NBEdge* otherConnFrom = i->getEdgeFrom()->getNBEdge();
                const int linkIndex = node->getConnectionIndex(otherConnFrom, i->getNBEdgeConnection());
                std::string responseString = node->getResponse(linkIndex);
                std::reverse(responseString.begin(), responseString.end());
                // determine the prohibition status
                bool foes = ((int)currentFoesString.size() > linkIndex) && (currentFoesString[linkIndex] == '1');
                bool forbids = ((int)responseString.size() > currentLinkIndex) && (responseString[currentLinkIndex] == '1');
                bool forbidden = ((int)currentResponseString.size() > linkIndex) && (currentResponseString[linkIndex] == '1');
                // insert in myConcernedConns
                myConcernedConns.insert(i);
                // change color depending of prohibition status
                if (!foes) {
                    i->setSpecialColor(&myLegend->getUndefinedColor());
                } else {
                    if (forbids && forbidden) {
                        i->setSpecialColor(&myLegend->getMutualConflictColor());
                    } else if (forbids) {
                        i->setSpecialColor(&myLegend->getProhibitedColor());
                    } else if (forbidden) {
                        i->setSpecialColor(&myLegend->getProhibitingColor());
                    } else {
                        i->setSpecialColor(&myLegend->getUnregulatedConflictColor());
                    }
                }
            }
        }
        // update description
        myRelativeToConnection->updateDescription();
    }
}


/****************************************************************************/
