/****************************************************************************/
// Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
// Copyright (C) 2006-2022 German Aerospace Center (DLR) and others.
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
/// @file    FXGroupBoxModule.cpp
/// @author  Pablo Alvarez Lopez
/// @date    Dec 2021
///
//
/****************************************************************************/

/* =========================================================================
 * included modules
 * ======================================================================= */
#include <config.h>

#include <utils/gui/div/GUIDesigns.h>
#include <utils/gui/images/GUIIconSubSys.h>
#include <utils/gui/windows/GUIAppEnum.h>
#include <netedit/frames/GNEFrame.h>
#include <netedit/GNEViewNet.h>
#include <netedit/GNEViewParent.h>

#include "FXGroupBoxModule.h"


// ===========================================================================
// FOX callback mapping
// ===========================================================================

FXDEFMAP(FXGroupBoxModule) FXGroupBoxModuleMap[] = {
    FXMAPFUNC(SEL_PAINT,    0,                              FXGroupBoxModule::onPaint),
    FXMAPFUNC(SEL_COMMAND,  MID_GROUPBOXMODULE_COLLAPSE,    FXGroupBoxModule::onCmdCollapseButton),
    FXMAPFUNC(SEL_COMMAND,  MID_GROUPBOXMODULE_EXTEND,      FXGroupBoxModule::onCmdExtendButton),
    FXMAPFUNC(SEL_COMMAND,  MID_GROUPBOXMODULE_RESETWIDTH,  FXGroupBoxModule::onCmdResetButton),
    FXMAPFUNC(SEL_COMMAND,  MID_GROUPBOXMODULE_SAVE,        FXGroupBoxModule::onCmdSaveButton),
    FXMAPFUNC(SEL_COMMAND,  MID_GROUPBOXMODULE_LOAD,        FXGroupBoxModule::onCmdLoadButton),
    FXMAPFUNC(SEL_UPDATE,   MID_GROUPBOXMODULE_RESETWIDTH,  FXGroupBoxModule::onUpdResetButton),
};

// Object implementation
FXIMPLEMENT(FXGroupBoxModule, FXVerticalFrame, FXGroupBoxModuleMap, ARRAYNUMBER(FXGroupBoxModuleMap))

// ===========================================================================
// method definitions
// ===========================================================================

FXGroupBoxModule::FXGroupBoxModule(GNEFrame* frame, const std::string& text, const int options) :
    FXVerticalFrame(frame->getContentFrame(), GUIDesignHorizontalFrame),
    myOptions(options),
    myFrameParent(frame),
    myCollapsed(false) {
    // build button and labels
    FXHorizontalFrame* headerFrame = new FXHorizontalFrame(this, GUIDesignAuxiliarHorizontalFrame);
    if (myOptions & Options::COLLAPSIBLE) {
        myCollapseButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::COLLAPSE), this, MID_GROUPBOXMODULE_COLLAPSE, GUIDesignButtonFXGroupBoxModule);
    }
    if (myOptions & Options::EXTENSIBLE) {
        myExtendButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::EXTEND), this, MID_GROUPBOXMODULE_EXTEND, GUIDesignButtonFXGroupBoxModule);
        myResetWidthButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::RESET), this, MID_GROUPBOXMODULE_RESETWIDTH, GUIDesignButtonFXGroupBoxModule);
    }
    if (myOptions & Options::SAVE) {
        mySaveButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::SAVE), this, MID_GROUPBOXMODULE_SAVE, GUIDesignButtonFXGroupBoxModule);
    }
    if (myOptions & Options::LOAD) {
        myLoadButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::OPEN_NET), this, MID_GROUPBOXMODULE_LOAD, GUIDesignButtonFXGroupBoxModule);
    }
    myLabel = new FXLabel(headerFrame, text.c_str(), nullptr, GUIDesignLabelFXGroupBoxModule);
    // build collapsable frame
    myCollapsableFrame = new FXVerticalFrame(this, GUIDesignCollapsableFrame);
}


FXGroupBoxModule::FXGroupBoxModule(FXVerticalFrame* contentFrame, const std::string& text, const int options) :
    FXVerticalFrame(contentFrame, GUIDesignHorizontalFrame),
    myOptions(options),
    myCollapsed(false) {
    // build button and labels
    FXHorizontalFrame* headerFrame = new FXHorizontalFrame(this, GUIDesignAuxiliarHorizontalFrame);
    if (myOptions & Options::COLLAPSIBLE) {
        myCollapseButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::COLLAPSE), this, MID_GROUPBOXMODULE_COLLAPSE, GUIDesignButtonFXGroupBoxModule);
    }
    if (myOptions & Options::EXTENSIBLE) {
        throw ProcessError("This FXGroupBoxModule doesn't support Extensible flag");
    }
    if (myOptions & Options::SAVE) {
        mySaveButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::SAVE), this, MID_GROUPBOXMODULE_SAVE, GUIDesignButtonFXGroupBoxModule);
    }
    if (myOptions & Options::LOAD) {
        myLoadButton = new FXButton(headerFrame, "", GUIIconSubSys::getIcon(GUIIcon::OPEN_NET), this, MID_GROUPBOXMODULE_LOAD, GUIDesignButtonFXGroupBoxModule);
    }
    myLabel = new FXLabel(headerFrame, text.c_str(), nullptr, GUIDesignLabelFXGroupBoxModule);
    // build collapsable frame
    myCollapsableFrame = new FXVerticalFrame(this, GUIDesignCollapsableFrame);
}


FXGroupBoxModule::~FXGroupBoxModule() {}


void
FXGroupBoxModule::setText(const std::string& text) {
    myLabel->setText(text.c_str());
}


FXVerticalFrame*
FXGroupBoxModule::getCollapsableFrame() {
    return myCollapsableFrame;
}


long
FXGroupBoxModule::onPaint(FXObject*, FXSelector, void* ptr) {
    FXEvent* event = (FXEvent*)ptr;
    FXDCWindow dc(this, event);
    // Paint background
    dc.setForeground(backColor);
    dc.fillRectangle(event->rect.x, event->rect.y, event->rect.w, event->rect.h);
    // draw groove rectangle
    drawGrooveRectangle(dc, 0, 15, width, height - 15);
    return 1;
}


long
FXGroupBoxModule::onCmdCollapseButton(FXObject*, FXSelector, void*) {
    if (myCollapsed) {
        myCollapsed = false;
        myCollapseButton->setIcon(GUIIconSubSys::getIcon(GUIIcon::COLLAPSE));
        myCollapsableFrame->show();
    } else {
        myCollapsed = true;
        myCollapseButton->setIcon(GUIIconSubSys::getIcon(GUIIcon::UNCOLLAPSE));
        myCollapsableFrame->hide();
    }
    recalc();
    return 1;
}


long 
FXGroupBoxModule::onCmdExtendButton(FXObject*, FXSelector, void*) {
    if (myFrameParent) {
        int maximumWidth = -1;
        // search in every child 
        for(auto child = getFirst(); child != nullptr; child = child->getNext()) {
            // check if child is an scrollWindow
            auto scrollWindow = dynamic_cast<FXScrollWindow*>(child->getFirst());
            if (scrollWindow && (scrollWindow->getContentWidth() > maximumWidth)) {
                maximumWidth = scrollWindow->getContentWidth();
            }
        }
        // now set parent parent width
        if (maximumWidth != -1) {
            myFrameParent->getViewNet()->getViewParent()->setFrameAreaWith(maximumWidth);
        }
    }
    return 1;
}


long 
FXGroupBoxModule::onCmdResetButton(FXObject*, FXSelector, void*) {
    if (myFrameParent) {
        myFrameParent->getViewNet()->getViewParent()->setFrameAreaWith(220);
    }
    return 1;
}


long 
FXGroupBoxModule::onUpdResetButton(FXObject* sender, FXSelector, void*) {
    if (myFrameParent) {
        if (myFrameParent->getViewNet()->getViewParent()->getFrameAreaWith() == 220) {
            sender->handle(this, FXSEL(SEL_COMMAND, ID_DISABLE), nullptr);
        } else {
            sender->handle(this, FXSEL(SEL_COMMAND, ID_ENABLE), nullptr);
        }
    }
    return 1;
}


long
FXGroupBoxModule::onCmdSaveButton(FXObject*, FXSelector, void*) {
    return saveContents();
}


long
FXGroupBoxModule::onCmdLoadButton(FXObject*, FXSelector, void*) {
    return loadContents();
}


FXGroupBoxModule::FXGroupBoxModule() :
    myOptions(Options::NOTHING),
    myCollapsed(false) {
}


bool
FXGroupBoxModule::saveContents() const {
    // nothing to do
    return false;
}


bool
FXGroupBoxModule::loadContents() const {
    // nothing to do
    return false;
}


void
FXGroupBoxModule::toogleSaveButton(const bool value) {
    if (mySaveButton) {
        if (value) {
            mySaveButton->enable();
        } else {
            mySaveButton->disable();
        }
    }
}
