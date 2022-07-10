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
/// @file    GNETLSEditorFrame.cpp
/// @author  Jakob Erdmann
/// @date    May 2011
///
// The Widget for modifying traffic lights
/****************************************************************************/
#include <config.h>

#include <netbuild/NBLoadedSUMOTLDef.h>
#include <netedit/GNENet.h>
#include <netedit/GNEUndoList.h>
#include <netedit/GNEViewNet.h>
#include <netedit/changes/GNEChange_TLS.h>
#include <netedit/dialogs/GNESingleParametersDialog.h>
#include <netedit/elements/network/GNEInternalLane.h>
#include <netimport/NIXMLTrafficLightsHandler.h>
#include <utils/gui/div/GUIDesigns.h>
#include <utils/gui/windows/GUIAppEnum.h>
#include <utils/options/OptionsCont.h>
#include <utils/xml/XMLSubSys.h>

#include "GNETLSEditorFrame.h"

// ===========================================================================
// FOX callback mapping
// ===========================================================================

FXDEFMAP(GNETLSEditorFrame) GNETLSEditorFrameMap[] = {
    FXMAPFUNC(SEL_COMMAND,    MID_CANCEL,                       GNETLSEditorFrame::onCmdCancel),
    FXMAPFUNC(SEL_UPDATE,     MID_CANCEL,                       GNETLSEditorFrame::onUpdModified),
    FXMAPFUNC(SEL_COMMAND,    MID_OK,                           GNETLSEditorFrame::onCmdOK),
    FXMAPFUNC(SEL_UPDATE,     MID_OK,                           GNETLSEditorFrame::onUpdModified),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_CREATE,          GNETLSEditorFrame::onCmdDefCreate),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_CREATE,          GNETLSEditorFrame::onUpdDefCreate),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_DELETE,          GNETLSEditorFrame::onCmdDefDelete),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_DELETE,          GNETLSEditorFrame::onUpdDefSwitch),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_REGENERATE,      GNETLSEditorFrame::onCmdDefRegenerate),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_REGENERATE,      GNETLSEditorFrame::onUpdDefSwitch),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_SWITCH,          GNETLSEditorFrame::onCmdDefSwitch),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_SWITCH,          GNETLSEditorFrame::onUpdDefSwitch),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_OFFSET,          GNETLSEditorFrame::onCmdSetOffset),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_OFFSET,          GNETLSEditorFrame::onUpdNeedsDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_PARAMETERS,      GNETLSEditorFrame::onCmdSetParameters),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_PARAMETERS,      GNETLSEditorFrame::onUpdNeedsDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_RENAME,          GNETLSEditorFrame::onCmdDefRename),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_SUBRENAME,       GNETLSEditorFrame::onCmdDefSubRename),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_ADDOFF,          GNETLSEditorFrame::onCmdDefAddOff),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_GUESSPROGRAM,    GNETLSEditorFrame::onCmdGuess),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_PHASE_CREATE,    GNETLSEditorFrame::onCmdPhaseCreate),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_PHASE_CREATE,    GNETLSEditorFrame::onUpdNeedsDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_PHASE_DELETE,    GNETLSEditorFrame::onCmdPhaseDelete),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_PHASE_DELETE,    GNETLSEditorFrame::onUpdNeedsDefAndPhase),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_CLEANUP,         GNETLSEditorFrame::onCmdCleanup),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_CLEANUP,         GNETLSEditorFrame::onUpdNeedsSingleDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_ADDUNUSED,       GNETLSEditorFrame::onCmdAddUnused),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_ADDUNUSED,       GNETLSEditorFrame::onUpdNeedsDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_GROUP_STATES,    GNETLSEditorFrame::onCmdGroupStates),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_GROUP_STATES,    GNETLSEditorFrame::onUpdNeedsSingleDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_UNGROUP_STATES,  GNETLSEditorFrame::onCmdUngroupStates),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_UNGROUP_STATES,  GNETLSEditorFrame::onUpdUngroupStates),
    FXMAPFUNC(SEL_SELECTED,   MID_GNE_TLSFRAME_PHASE_TABLE,     GNETLSEditorFrame::onCmdPhaseSwitch),
    FXMAPFUNC(SEL_DESELECTED, MID_GNE_TLSFRAME_PHASE_TABLE,     GNETLSEditorFrame::onCmdPhaseSwitch),
    FXMAPFUNC(SEL_CHANGED,    MID_GNE_TLSFRAME_PHASE_TABLE,     GNETLSEditorFrame::onCmdPhaseSwitch),
    FXMAPFUNC(SEL_REPLACED,   MID_GNE_TLSFRAME_PHASE_TABLE,     GNETLSEditorFrame::onCmdPhaseEdit),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_OPEN_PARAMETERS_DIALOG,   GNETLSEditorFrame::onCmdEditParameters),
};

FXDEFMAP(GNETLSEditorFrame::TLSFile) TLSFileMap[] = {
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_LOAD_PROGRAM,    GNETLSEditorFrame::TLSFile::onCmdLoadTLSProgram),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_LOAD_PROGRAM,    GNETLSEditorFrame::TLSFile::onUpdNeedsDef),
    FXMAPFUNC(SEL_COMMAND,    MID_GNE_TLSFRAME_SAVE_PROGRAM,    GNETLSEditorFrame::TLSFile::onCmdSaveTLSProgram),
    FXMAPFUNC(SEL_UPDATE,     MID_GNE_TLSFRAME_SAVE_PROGRAM,    GNETLSEditorFrame::TLSFile::onUpdNeedsDef),
};

// Object implementation
FXIMPLEMENT(GNETLSEditorFrame,          FXVerticalFrame,    GNETLSEditorFrameMap,   ARRAYNUMBER(GNETLSEditorFrameMap))
FXIMPLEMENT(GNETLSEditorFrame::TLSFile, FXGroupBoxModule,         TLSFileMap,             ARRAYNUMBER(TLSFileMap))


// ===========================================================================
// method definitions
// ===========================================================================

GNETLSEditorFrame::GNETLSEditorFrame(FXHorizontalFrame* horizontalFrameParent, GNEViewNet* viewNet):
    GNEFrame(horizontalFrameParent, viewNet, "Edit Traffic Light"),
    myEditedDef(nullptr) {

    // Create Overlapped Inspection modul
    myOverlappedInspection = new GNEOverlappedInspection(this, SUMO_TAG_JUNCTION);

    // create TLSJunction modul
    myTLSJunction = new GNETLSEditorFrame::TLSJunction(this);

    // create TLSDefinition modul
    myTLSDefinition = new GNETLSEditorFrame::TLSDefinition(this);

    // create TLSAttributes modul
    myTLSAttributes = new GNETLSEditorFrame::TLSAttributes(this);

    // create TLSModifications modul
    myTLSModifications = new GNETLSEditorFrame::TLSModifications(this);

    // create TLSPhases modul
    myTLSPhases = new GNETLSEditorFrame::TLSPhases(this);

    // create TLSFile modul
    myTLSFile = new GNETLSEditorFrame::TLSFile(this);

    // "Add 'off' program"
    /*
    new FXButton(myContentFrame, "Add \"Off\"-Program\t\tAdds a program for switching off this traffic light",
            0, this, MID_GNE_TLSFRAME_ADDOFF, GUIDesignButton);
    */
}


GNETLSEditorFrame::~GNETLSEditorFrame() {
    cleanup();
}


void
GNETLSEditorFrame::show() {
    // hide myOverlappedInspection
    myOverlappedInspection->hideOverlappedInspection();
    GNEFrame::show();
}

void
GNETLSEditorFrame::editTLS(const Position& clickedPosition, const GNEViewNetHelper::ObjectsUnderCursor& objectsUnderCursor) {
    // first check if in objectsUnderCursor there is a junction
    if (objectsUnderCursor.getJunctionFront()) {
        // show objects under cursor
        myOverlappedInspection->showOverlappedInspection(objectsUnderCursor, clickedPosition);
        // hide if we inspect only one junction
        if (myOverlappedInspection->getNumberOfOverlappedACs() == 1) {
            myOverlappedInspection->hideOverlappedInspection();
        }
        // set junction
        editJunction(objectsUnderCursor.getJunctionFront());
    } else {
        myViewNet->setStatusBarText("Click over a junction to edit a TLS");
    }
}


bool
GNETLSEditorFrame::isTLSSaved() {
    if (myTLSModifications->checkHaveModifications()) {
        // write warning if netedit is running in testing mode
        WRITE_DEBUG("Opening question FXMessageBox 'save TLS'");
        // open question box
        FXuint answer = FXMessageBox::question(this, MBOX_YES_NO_CANCEL,
                                               "Save TLS Changes", "%s",
                                               "There is unsaved changes in current edited traffic light.\nDo you want to save it before changing mode?");
        if (answer == MBOX_CLICKED_YES) { //1:yes, 2:no, 4:esc/cancel
            // write warning if netedit is running in testing mode
            WRITE_DEBUG("Closed FXMessageBox 'save TLS' with 'YES'");
            // save modifications
            onCmdOK(nullptr, 0, nullptr);
            return true;
        } else if (answer == MBOX_CLICKED_NO) {
            // write warning if netedit is running in testing mode
            WRITE_DEBUG("Closed FXMessageBox 'save TLS' with 'No'");
            // cancel modifications
            onCmdCancel(nullptr, 0, nullptr);
            return true;
        } else {
            // write warning if netedit is running in testing mode
            WRITE_DEBUG("Closed FXMessageBox 'save TLS' with 'Cancel'");
            // abort changing mode
            return false;
        }
    } else {
        return true;
    }
}


bool
GNETLSEditorFrame::parseTLSPrograms(const std::string& file) {
    NBTrafficLightLogicCont& tllCont = myViewNet->getNet()->getTLLogicCont();
    NBTrafficLightLogicCont tmpTLLCont;
    NIXMLTrafficLightsHandler tllHandler(tmpTLLCont, myViewNet->getNet()->getEdgeCont());
    // existing definitions must be available to update their programs
    std::set<NBTrafficLightDefinition*> origDefs;
    for (NBTrafficLightDefinition* def : tllCont.getDefinitions()) {
        // make a copy of every program
        NBTrafficLightLogic* logic = tllCont.getLogic(def->getID(), def->getProgramID());
        if (logic != nullptr) {
            NBTrafficLightDefinition* copy = new NBLoadedSUMOTLDef(*def, *logic);
            std::vector<NBNode*> nodes = def->getNodes();
            for (auto it_node : nodes) {
                GNEJunction* junction = myViewNet->getNet()->getAttributeCarriers()->retrieveJunction(it_node->getID());
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, def, false, false), true);
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, copy, true), true);
            }
            tmpTLLCont.insert(copy);
            origDefs.insert(copy);
        } else {
            WRITE_WARNING("tlLogic '" + def->getID() + "', program '" + def->getProgramID() + "' could not be built");
        }
    }
    //std::cout << " initialized tmpCont with " << origDefs.size() << " defs\n";
    XMLSubSys::runParser(tllHandler, file);

    std::vector<NBLoadedSUMOTLDef*> loadedTLS;
    for (NBTrafficLightDefinition* def : tmpTLLCont.getDefinitions()) {
        NBLoadedSUMOTLDef* sdef = dynamic_cast<NBLoadedSUMOTLDef*>(def);
        if (sdef != nullptr) {
            loadedTLS.push_back(sdef);
        }
    }
    myViewNet->setStatusBarText("Loaded " + toString(loadedTLS.size()) + " programs");
    for (auto def : loadedTLS) {
        if (origDefs.count(def) != 0) {
            // already add to undolist before
            //std::cout << " skip " << def->getDescription() << "\n";
            continue;
        }
        std::vector<NBNode*> nodes = def->getNodes();
        //std::cout << " add " << def->getDescription() << " for nodes=" << toString(nodes) << "\n";
        for (auto it_node : nodes) {
            GNEJunction* junction = myViewNet->getNet()->getAttributeCarriers()->retrieveJunction(it_node->getID());
            //myViewNet->getUndoList()->add(new GNEChange_TLS(junction, myTLSEditorParent->myEditedDef, false), true);
            myViewNet->getUndoList()->add(new GNEChange_TLS(junction, def, true), true);
        }
    }
    // clean up temporary container to avoid deletion of defs when it's destruct is called
    for (NBTrafficLightDefinition* def : tmpTLLCont.getDefinitions()) {
        tmpTLLCont.removeProgram(def->getID(), def->getProgramID(), false);
    }
    return true;
}


long
GNETLSEditorFrame::onCmdCancel(FXObject*, FXSelector, void*) {
    if (myTLSJunction->getCurrentJunction() != nullptr) {
        myViewNet->getUndoList()->abortAllChangeGroups();
        cleanup();
        myViewNet->updateViewNet();
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdOK(FXObject*, FXSelector, void*) {
    if (myTLSJunction->getCurrentJunction() != nullptr) {
        if (myTLSModifications->checkHaveModifications()) {
            NBTrafficLightDefinition* oldDefinition = myTLSAttributes->getCurrentTLSDefinition();
            std::vector<NBNode*> nodes = oldDefinition->getNodes();
            for (const auto &node : nodes) {
                GNEJunction* junction = myViewNet->getNet()->getAttributeCarriers()->retrieveJunction(node->getID());
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, oldDefinition, false), true);
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, myEditedDef, true), true);
            }
            myEditedDef = nullptr;
            myViewNet->getUndoList()->end();
            cleanup();
            myViewNet->updateViewNet();
        } else {
            onCmdCancel(nullptr, 0, nullptr);
        }
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdDefCreate(FXObject*, FXSelector, void*) {
    GNEJunction* junction = myTLSJunction->getCurrentJunction();
    // get current TLS program id
    const auto currentTLS = myTLSAttributes->getCurrentTLSProgramID();
    // abort because we onCmdOk assumes we wish to save an edited definition
    onCmdCancel(nullptr, 0, nullptr);
    // check that current junction has two or more edges
    if ((junction->getGNEIncomingEdges().size() > 0) && (junction->getGNEOutgoingEdges().size() > 0)) {
        if (junction->getAttribute(SUMO_ATTR_TYPE) != toString(SumoXMLNodeType::TRAFFIC_LIGHT)) {
            junction->setAttribute(SUMO_ATTR_TYPE, toString(SumoXMLNodeType::TRAFFIC_LIGHT), myViewNet->getUndoList());
        } else {
            if (junction->getNBNode()->isTLControlled()) {
                // use existing traffic light as template for type, signal groups, controlled nodes etc
                NBTrafficLightDefinition* tpl = nullptr;
                for (const auto& TLS : junction->getNBNode()->getControllingTLS()) {
                    if (TLS->getProgramID() == currentTLS) {
                        tpl = TLS;
                    }
                }
                if (tpl == nullptr) {
                    tpl = *junction->getNBNode()->getControllingTLS().begin();
                }
                NBTrafficLightLogic* newLogic = tpl->compute(OptionsCont::getOptions());
                NBLoadedSUMOTLDef* newDef = new NBLoadedSUMOTLDef(*tpl, *newLogic);
                delete newLogic;
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, newDef, true, true), true);
            } else {
                // for some reason the traffic light was not built, try again
                myViewNet->getUndoList()->add(new GNEChange_TLS(junction, nullptr, true, true), true);
            }
        }
        editJunction(junction);
    } else {
        // write warning if netedit is running in testing mode
        WRITE_DEBUG("Opening warning FXMessageBox 'invalid TLS'");
        // open question box
        FXMessageBox::warning(this, MBOX_OK,
                              "TLS cannot be created", "%s",
                              "Traffic Light cannot be created because junction must have\n at least one incoming edge and one outgoing edge.");
        // write warning if netedit is running in testing mode
        WRITE_DEBUG("Closed FXMessageBox 'invalid TLS'");
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdDefDelete(FXObject*, FXSelector, void*) {
    GNEJunction* junction = myTLSJunction->getCurrentJunction();
    const bool changeType = myTLSAttributes->getNumberOfTLSDefinitions() == 1;
    NBTrafficLightDefinition* tlDef = myTLSAttributes->getCurrentTLSDefinition();
    onCmdCancel(nullptr, 0, nullptr); // abort because onCmdOk assumes we wish to save an edited definition
    if (changeType) {
        junction->setAttribute(SUMO_ATTR_TYPE, toString(SumoXMLNodeType::PRIORITY), myViewNet->getUndoList());
    } else {
        myViewNet->getUndoList()->add(new GNEChange_TLS(junction, tlDef, false), true);
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdDefRegenerate(FXObject*, FXSelector, void*) {
    // make a copy of the junction
    GNEJunction* junction = myTLSJunction->getCurrentJunction();
    // begin undo
    myViewNet->getUndoList()->begin(GUIIcon::MODETLS, "regenerate TLS");
    // delete junction
    onCmdDefDelete(nullptr, 0, nullptr);
    // set junction again
    myTLSJunction->setCurrentJunction(junction);
    // create junction
    onCmdDefCreate(nullptr, 0, nullptr);
    // end undo
    myViewNet->getUndoList()->end();
    return 1;
}


long
GNETLSEditorFrame::onCmdDefSwitch(FXObject*, FXSelector, void*) {
    assert(myTLSJunction->getCurrentJunction() != 0);
    assert(myTLSAttributes->getNumberOfTLSDefinitions() == myTLSAttributes->getNumberOfPrograms());
    NBTrafficLightDefinition* tlDef = myTLSAttributes->getCurrentTLSDefinition();
    // logic may not have been recomputed yet. recompute to be sure
    NBTrafficLightLogicCont& tllCont = myViewNet->getNet()->getTLLogicCont();
    myViewNet->getNet()->computeJunction(myTLSJunction->getCurrentJunction());
    NBTrafficLightLogic* tllogic = tllCont.getLogic(tlDef->getID(), tlDef->getProgramID());
    if (tllogic != nullptr) {
        // now we can be sure that the tlDef is up to date (i.e. re-guessed)
        buildInternalLanes(tlDef);
        // create working copy from original def
        delete myEditedDef;
        myEditedDef = new NBLoadedSUMOTLDef(*tlDef, *tllogic);
        myTLSAttributes->setOffset(myEditedDef->getLogic()->getOffset());
        myTLSAttributes->setParameters(myEditedDef->getLogic()->getParametersStr());
        myTLSPhases->initPhaseTable();
        myTLSPhases->updateCycleDuration();
        myTLSPhases->showCycleDuration();
    } else {
        // tlDef has no valid logic (probably because id does not control any links
        onCmdCancel(nullptr, 0, nullptr);
        myViewNet->setStatusBarText("Traffic light does not control any links");
    }
    return 1;
}


long
GNETLSEditorFrame::onUpdDefSwitch(FXObject* o, FXSelector, void*) {
    const bool enable = myTLSAttributes->getNumberOfTLSDefinitions() > 0 && !myTLSModifications->checkHaveModifications();
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


long
GNETLSEditorFrame::onUpdNeedsDef(FXObject* o, FXSelector, void*) {
    const bool enable = myTLSAttributes->getNumberOfTLSDefinitions() > 0;
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


long
GNETLSEditorFrame::onUpdNeedsDefAndPhase(FXObject* o, FXSelector, void*) {
    // do not delete the last phase
    const bool enable = myTLSAttributes->getNumberOfTLSDefinitions() > 0 && myTLSPhases->getPhaseTable()->getNumRows() > 1;
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


long
GNETLSEditorFrame::onUpdDefCreate(FXObject* o, FXSelector, void*) {
    GNEJunction* junction = myTLSJunction->getCurrentJunction();
    const bool enable = junction != nullptr && !myTLSModifications->checkHaveModifications();
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    const bool copy = junction != nullptr && junction->getNBNode()->isTLControlled();
    static_cast<FXButton*>(o)->setText(copy ? "Copy" : "Create");
    return 1;
}


long
GNETLSEditorFrame::onUpdModified(FXObject* o, FXSelector, void*) {
    bool enable = myTLSModifications->checkHaveModifications();
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}



long
GNETLSEditorFrame::onCmdSetOffset(FXObject*, FXSelector, void*) {
    if (myTLSAttributes->isValidOffset()) {
        myTLSModifications->setHaveModifications(true);
        myEditedDef->setOffset(myTLSAttributes->getOffset());
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdSetParameters(FXObject*, FXSelector, void*) {
    if (myTLSAttributes->isValidParameters()) {
        myTLSModifications->setHaveModifications(true);
        myEditedDef->setParametersStr(myTLSAttributes->getParameters());
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdDefRename(FXObject*, FXSelector, void*) {
    return 1;
}


long
GNETLSEditorFrame::onCmdDefSubRename(FXObject*, FXSelector, void*) {
    return 1;
}


long
GNETLSEditorFrame::onCmdDefAddOff(FXObject*, FXSelector, void*) {
    return 1;
}


long
GNETLSEditorFrame::onCmdGuess(FXObject*, FXSelector, void*) {
    return 1;
}


long
GNETLSEditorFrame::onCmdPhaseSwitch(FXObject*, FXSelector, void*) {
    const int index = myTLSPhases->getPhaseTable()->getCurrentRow();
    const NBTrafficLightLogic::PhaseDefinition& phase = getPhases()[index];
    myTLSPhases->getPhaseTable()->selectRow(index);
    // need not hold since links could have been deleted somewhere else and indices may be reused
    // assert(phase.state.size() == myInternalLanes.size());
    for (auto it : myInternalLanes) {
        int tlIndex = it.first;
        std::vector<GNEInternalLane*> lanes = it.second;
        LinkState state = LINKSTATE_DEADEND;
        if (tlIndex >= 0 && tlIndex < (int)phase.state.size()) {
            state = (LinkState)phase.state[tlIndex];
        }
        for (auto it_lane : lanes) {
            it_lane->setLinkState(state);
        }
    }
    myViewNet->updateViewNet();
    return 1;
}


void
GNETLSEditorFrame::selectedOverlappedElement(GNEAttributeCarrier* AC) {
    editJunction(dynamic_cast<GNEJunction*>(AC));
}


long
GNETLSEditorFrame::onCmdPhaseCreate(FXObject*, FXSelector, void*) {
    // mark TLS as modified
    myTLSModifications->setHaveModifications(true);
    // check if TLS is static
    const bool TLSStatic = (myEditedDef->getType() == TrafficLightType::STATIC);
    // allows insertion at first position by deselecting via arrow keys
    int newIndex = myTLSPhases->getPhaseTable()->getSelStartRow() + 1;
    int oldIndex = MAX2(0, myTLSPhases->getPhaseTable()->getSelStartRow());
    // copy current row
    SUMOTime duration = getSUMOTime(myTLSPhases->getPhaseTable()->getItemText(oldIndex, 0).text());
    const std::string oldState = myTLSPhases->getPhaseTable()->getItemText(oldIndex, TLSStatic? 1 : 3).text();
    std::string state = oldState;

    std::set<int> crossingIndices;
    for (NBNode* n : myEditedDef->getNodes()) {
        for (NBNode::Crossing* c : n->getCrossings()) {
            crossingIndices.insert(c->tlLinkIndex);
            crossingIndices.insert(c->tlLinkIndex2);
        }
    }

    // smart adapations for new state
    bool haveGreen = false;
    bool haveYellow = false;
    for (char c : state) {
        if (c == LINKSTATE_TL_GREEN_MAJOR || c == LINKSTATE_TL_GREEN_MINOR) {
            haveGreen = true;
        } else if (c == LINKSTATE_TL_YELLOW_MAJOR || c == LINKSTATE_TL_YELLOW_MINOR) {
            haveYellow = true;
        }
    }
    const OptionsCont& oc = OptionsCont::getOptions();
    if (haveGreen && haveYellow) {
        // guess left-mover state
        duration = TIME2STEPS(oc.getInt("tls.left-green.time"));
        for (int i = 0; i < (int)state.size(); i++) {
            if (state[i] == LINKSTATE_TL_YELLOW_MAJOR || state[i] == LINKSTATE_TL_YELLOW_MINOR) {
                state[i] = LINKSTATE_TL_RED;
            } else if (state[i] == LINKSTATE_TL_GREEN_MINOR) {
                state[i] = LINKSTATE_TL_GREEN_MAJOR;
            }
        }
    } else if (haveGreen) {
        // guess yellow state
        myEditedDef->setParticipantsInformation();
        duration = TIME2STEPS(myEditedDef->computeBrakingTime(oc.getFloat("tls.yellow.min-decel")));
        for (int i = 0; i < (int)state.size(); i++) {
            if (state[i] == LINKSTATE_TL_GREEN_MAJOR || state[i] == LINKSTATE_TL_GREEN_MINOR) {
                if (crossingIndices.count(i) == 0) {
                    state[i] = LINKSTATE_TL_YELLOW_MINOR;
                } else {
                    state[i] = LINKSTATE_TL_RED;
                }
            }
        }
    } else if (haveYellow) {
        duration = TIME2STEPS(oc.isDefault("tls.allred.time") ? 2 :  oc.getInt("tls.allred.time"));
        // guess all-red state
        for (int i = 0; i < (int)state.size(); i++) {
            if (state[i] == LINKSTATE_TL_YELLOW_MAJOR || state[i] == LINKSTATE_TL_YELLOW_MINOR) {
                state[i] = LINKSTATE_TL_RED;
            }
        }
    }
    // fix continuous green states
    const int nextIndex = myTLSPhases->getPhaseTable()->getNumRows() > newIndex ? newIndex : 0;
    const std::string state2 = myTLSPhases->getPhaseTable()->getItemText(nextIndex, TLSStatic? 1 : 3).text();
    for (int i = 0; i < (int)state.size(); i++) {
        if ((oldState[i] == LINKSTATE_TL_GREEN_MAJOR || oldState[i] == LINKSTATE_TL_GREEN_MINOR)
                && (state2[i] == LINKSTATE_TL_GREEN_MAJOR || state2[i] == LINKSTATE_TL_GREEN_MINOR)) {
            state[i] = oldState[i];
        }
    }

    myEditedDef->getLogic()->addStep(duration, state, std::vector<int>(), "", newIndex);
    myTLSPhases->getPhaseTable()->setCurrentItem(newIndex, 0);
    myTLSPhases->initPhaseTable(newIndex);
    myTLSPhases->updateCycleDuration();
    myTLSPhases->getPhaseTable()->setFocus();
    return 1;
}


long
GNETLSEditorFrame::onCmdPhaseDelete(FXObject*, FXSelector, void*) {
    myTLSModifications->setHaveModifications(true);
    const int newRow = MAX2((int)0, (int)myTLSPhases->getPhaseTable()->getCurrentRow() - 1);
    myEditedDef->getLogic()->deletePhase(myTLSPhases->getPhaseTable()->getCurrentRow());
    myTLSPhases->initPhaseTable(newRow);
    myTLSPhases->updateCycleDuration();
    myTLSPhases->getPhaseTable()->setFocus();
    return 1;
}


long
GNETLSEditorFrame::onCmdCleanup(FXObject*, FXSelector, void*) {
    myTLSModifications->setHaveModifications(myEditedDef->cleanupStates());
    buildInternalLanes(myEditedDef);
    myTLSPhases->initPhaseTable(0);
    myTLSPhases->getPhaseTable()->setFocus();
    myTLSModifications->setHaveModifications(true);
    return 1;
}


long
GNETLSEditorFrame::onCmdAddUnused(FXObject*, FXSelector, void*) {
    myEditedDef->getLogic()->setStateLength(
        myEditedDef->getLogic()->getNumLinks() + 1);
    myTLSModifications->setHaveModifications(true);
    myTLSPhases->initPhaseTable(0);
    myTLSPhases->getPhaseTable()->setFocus();
    return 1;
}


long
GNETLSEditorFrame::onCmdGroupStates(FXObject*, FXSelector, void*) {
    myEditedDef->groupSignals();
    myTLSModifications->setHaveModifications(true);
    buildInternalLanes(myEditedDef);
    myTLSPhases->initPhaseTable(0);
    myTLSPhases->getPhaseTable()->setFocus();
    return 1;
}


long
GNETLSEditorFrame::onCmdUngroupStates(FXObject*, FXSelector, void*) {
    myEditedDef->setParticipantsInformation();
    myEditedDef->ungroupSignals();
    myTLSModifications->setHaveModifications(true);
    buildInternalLanes(myEditedDef);
    myTLSPhases->initPhaseTable(0);
    myTLSPhases->getPhaseTable()->setFocus();
    return 1;
}


long
GNETLSEditorFrame::onUpdNeedsSingleDef(FXObject* o, FXSelector, void*) {
    const bool enable = myTLSAttributes->getNumberOfTLSDefinitions() == 1;
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


long
GNETLSEditorFrame::onUpdUngroupStates(FXObject* o, FXSelector, void*) {
    const bool enable = myTLSAttributes->getNumberOfTLSDefinitions() == 1 && myEditedDef != nullptr && myEditedDef->usingSignalGroups();
    o->handle(this, FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


long
GNETLSEditorFrame::onCmdPhaseEdit(FXObject*, FXSelector, void* ptr) {
    /* @note: there is a bug when copying/pasting rows: when this handler is
     * called the value of the cell is not yet updated. This means you have to
     * click inside the cell and hit enter to actually update the value */
    FXTablePos* tp = (FXTablePos*)ptr;
    const std::string value = myTLSPhases->getPhaseTable()->getItemText(tp->row, tp->col).text();
    // Declare columns
    int colDuration = 0;
    int colState = -1;
    int colNext = -1;
    int colName = -1;
    int colMinDur = -1;
    int colMaxDur = -1;
    int colEarliestEnd = -1;
    int colLatestEnd = -1;
    int colVehExt = -1;
    int colYellow = -1;
    int colRed = -1;
    // set columns
    if (myEditedDef->getType() == TrafficLightType::STATIC) {
        colState = 1;
        colNext = 2;
        colName = 3;
    } else if (myEditedDef->getType() == TrafficLightType::ACTUATED) {
        colMinDur = 1;
        colMaxDur = 2;
        colState = 3;
        colEarliestEnd = 4;
        colLatestEnd = 5;
        colNext = 6;
        colName = 7;
    } else if (myEditedDef->getType() == TrafficLightType::DELAYBASED) {
        colMinDur = 1;
        colMaxDur = 2;
        colState = 3;
        colNext = 4;
    } else if (myEditedDef->getType() == TrafficLightType::NEMA) {
        colMinDur = 1;
        colMaxDur = 2;
        colState = 3;
        colVehExt = 4;
        colYellow = 5;
        colRed = 6;
        colNext = 7;
        colName = 8;
    }
    // check column
    if (tp->col == colDuration) {
        // duration edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime duration = getSUMOTime(value);
            if (duration > 0) {
                myEditedDef->getLogic()->setPhaseDuration(tp->row, duration);
                myTLSModifications->setHaveModifications(true);
                myTLSPhases->updateCycleDuration();
                return 1;
            }
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colDuration, getSteps2Time(getPhases()[tp->row].duration).c_str());
    } else if (tp->col == colState) {
        // state edited
        try {
            // insert phase with new step and delete the old phase
            const NBTrafficLightLogic::PhaseDefinition& phase = getPhases()[tp->row];
            myEditedDef->getLogic()->addStep(phase.duration, value, phase.next, phase.name, tp->row);
            myEditedDef->getLogic()->deletePhase(tp->row + 1);
            myTLSModifications->setHaveModifications(true);
            onCmdPhaseSwitch(nullptr, 0, nullptr);
        } catch (ProcessError&) {
            // input error, reset value
            myTLSPhases->getPhaseTable()->setItemText(tp->row, colState, getPhases()[tp->row].state.c_str());
        }
    } else if (tp->col == colNext) {
        // next edited
        bool ok = true;
        if (GNEAttributeCarrier::canParse<std::vector<int> >(value)) {
            std::vector<int> nextEdited = GNEAttributeCarrier::parse<std::vector<int> >(value);
            for (int n : nextEdited) {
                if (n < 0 || n >= myTLSPhases->getPhaseTable()->getNumRows()) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                myEditedDef->getLogic()->setPhaseNext(tp->row, nextEdited);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colNext, "");
    } else if (tp->col == colName) {
        // name edited
        myEditedDef->getLogic()->setPhaseName(tp->row, value);
        myTLSModifications->setHaveModifications(true);
    } else if (tp->col == colMinDur) {
        // minDur edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime minDur = getSUMOTime(value);
            if (minDur > 0) {
                myEditedDef->getLogic()->setPhaseMinDuration(tp->row, minDur);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseMinDuration(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colMinDur, varDurString(getPhases()[tp->row].minDur).c_str());
    } else if (tp->col == colMaxDur) {
        // maxDur edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime maxDur = getSUMOTime(value);
            if (maxDur > 0) {
                myEditedDef->getLogic()->setPhaseMaxDuration(tp->row, maxDur);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseMaxDuration(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colMaxDur, varDurString(getPhases()[tp->row].maxDur).c_str());
    } else if (tp->col == colEarliestEnd) {
        // earliestEnd edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime earliestEnd = getSUMOTime(value);
            if (earliestEnd > 0) {
                myEditedDef->getLogic()->setPhaseEarliestEnd(tp->row, earliestEnd);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseEarliestEnd(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colEarliestEnd, varDurString(getPhases()[tp->row].earliestEnd).c_str());
    } else if (tp->col == colLatestEnd) {
        // latestEnd edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime latestEnd = getSUMOTime(value);
            if (latestEnd > 0) {
                myEditedDef->getLogic()->setPhaseLatestEnd(tp->row, latestEnd);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseLatestEnd(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colLatestEnd, varDurString(getPhases()[tp->row].latestEnd).c_str());
    } else if (tp->col == colName) {
        // name edited
        myEditedDef->getLogic()->setPhaseName(tp->row, value);
        myTLSModifications->setHaveModifications(true);
    } else if (tp->col == colVehExt) {
        // vehExt edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime vehExt = getSUMOTime(value);
            if (vehExt > 0) {
                myEditedDef->getLogic()->setPhaseVehExt(tp->row, vehExt);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseVehExt(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colVehExt, varDurString(getPhases()[tp->row].vehExt).c_str());
    } else if (tp->col == colYellow) {
        // yellow edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime yellow = getSUMOTime(value);
            if (yellow > 0) {
                myEditedDef->getLogic()->setPhaseYellow(tp->row, yellow);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseYellow(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colYellow, varDurString(getPhases()[tp->row].yellow).c_str());
    } else if (tp->col == colRed) {
        // red edited
        if (GNEAttributeCarrier::canParse<double>(value)) {
            SUMOTime red = getSUMOTime(value);
            if (red > 0) {
                myEditedDef->getLogic()->setPhaseRed(tp->row, red);
                myTLSModifications->setHaveModifications(true);
                return 1;
            }
        } else if (StringUtils::prune(value).empty()) {
            myEditedDef->getLogic()->setPhaseRed(tp->row, NBTrafficLightDefinition::UNSPECIFIED_DURATION);
            myTLSModifications->setHaveModifications(true);
            return 1;
        }
        // input error, reset value
        myTLSPhases->getPhaseTable()->setItemText(tp->row, colRed, varDurString(getPhases()[tp->row].red).c_str());
    }
    return 1;
}


long
GNETLSEditorFrame::onCmdEditParameters(FXObject*, FXSelector, void*) {
    // continue depending of myEditedDef
    if (myEditedDef) {
        // get previous parameters
        const auto previousParameters = myTLSAttributes->getParameters();
        // write debug information
        WRITE_DEBUG("Open single parameters dialog");
        if (GNESingleParametersDialog(myViewNet->getApp(), myEditedDef).execute()) {
            // write debug information
            WRITE_DEBUG("Close single parameters dialog");
            // set parameters in textfield
            myTLSAttributes->setParameters(myEditedDef->getParametersStr());
            // only mark as modified if parameters are different
            if (myTLSAttributes->getParameters() != previousParameters) {
                myTLSModifications->setHaveModifications(true);
            }
        } else {
            // write debug information
            WRITE_DEBUG("Cancel single parameters dialog");
        }
    }
    return 1;
}


void
GNETLSEditorFrame::cleanup() {
    if (myTLSJunction->getCurrentJunction()) {
        myTLSJunction->getCurrentJunction()->selectTLS(false);
        if (myTLSAttributes->getNumberOfTLSDefinitions() > 0) {
            for (NBNode* node : myTLSAttributes->getCurrentTLSDefinition()->getNodes()) {
                myViewNet->getNet()->getAttributeCarriers()->retrieveJunction(node->getID())->selectTLS(false);
            }
        }
    }
    // clean data structures
    myTLSJunction->setCurrentJunction(nullptr);
    myTLSModifications->setHaveModifications(false);
    delete myEditedDef;
    myEditedDef = nullptr;
    // clear internal lanes
    buildInternalLanes(nullptr);
    // clean up controls
    myTLSAttributes->clearTLSAttributes();
    // only clears when there are no definitions
    myTLSPhases->initPhaseTable();
    myTLSPhases->hideCycleDuration();
    myTLSJunction->updateJunctionDescription();
}


void
GNETLSEditorFrame::buildInternalLanes(NBTrafficLightDefinition* tlDef) {
    // clean up previous internal lanes
    for (const auto& internalLanes : myInternalLanes) {
        for (const auto& internalLane : internalLanes.second) {
            // remove internal lane from grid
            myViewNet->getNet()->getGrid().removeAdditionalGLObject(internalLane);
            // delete internal lane
            delete internalLane;
        }
    }
    // clear container
    myInternalLanes.clear();
    // create new internal lanes
    if (tlDef != nullptr) {
        const int NUM_POINTS = 10;
        const NBNode* nbnCurrentJunction = myTLSJunction->getCurrentJunction()->getNBNode();
        // get innerID NWWriter_SUMO::writeInternalEdges
        const std::string innerID = ":" + nbnCurrentJunction->getID();
        const NBConnectionVector& links = tlDef->getControlledLinks();
        // iterate over links
        for (const auto& link : links) {
            int tlIndex = link.getTLIndex();
            PositionVector shape;
            try {
                const NBEdge::Connection& con = link.getFrom()->getConnection(link.getFromLane(), link.getTo(), link.getToLane());
                shape = con.shape;
                shape.append(con.viaShape);
            } catch (ProcessError&) {
                shape = link.getFrom()->getToNode()->computeInternalLaneShape(link.getFrom(), NBEdge::Connection(link.getFromLane(),
                        link.getTo(), link.getToLane()), NUM_POINTS);
            }
            if (shape.length() < 2) {
                // enlarge shape to ensure visibility
                shape.clear();
                const PositionVector laneShapeFrom = link.getFrom()->getLaneShape(link.getFromLane());
                const PositionVector laneShapeTo = link.getTo()->getLaneShape(link.getToLane());
                shape.push_back(laneShapeFrom.positionAtOffset(MAX2(0.0, laneShapeFrom.length() - 1)));
                shape.push_back(laneShapeTo.positionAtOffset(MIN2(1.0, laneShapeFrom.length())));
            }
            GNEInternalLane* internalLane = new GNEInternalLane(this, myTLSJunction->getCurrentJunction(), innerID + '_' + toString(tlIndex),  shape, tlIndex);
            // due GNEInternalLane aren't attribute carriers, we need to use the net grid
            myViewNet->getNet()->getGrid().addAdditionalGLObject(internalLane);
            myInternalLanes[tlIndex].push_back(internalLane);
        }
        // iterate over crossings
        for (const auto& nbn : tlDef->getNodes()) {
            for (const auto& crossing : nbn->getCrossings()) {
                if (crossing->tlLinkIndex2 > 0 && crossing->tlLinkIndex2 != crossing->tlLinkIndex) {
                    // draw both directions
                    PositionVector forward = crossing->shape;
                    forward.move2side(crossing->width / 4);
                    GNEInternalLane* internalLane = new GNEInternalLane(this, myTLSJunction->getCurrentJunction(), crossing->id, forward, crossing->tlLinkIndex);
                    // due GNEInternalLane aren't attribute carriers, we need to use the net grid
                    myViewNet->getNet()->getGrid().addAdditionalGLObject(internalLane);
                    myInternalLanes[crossing->tlLinkIndex].push_back(internalLane);
                    PositionVector backward = crossing->shape.reverse();
                    backward.move2side(crossing->width / 4);
                    GNEInternalLane* internalLaneReverse = new GNEInternalLane(this, myTLSJunction->getCurrentJunction(), crossing->id + "_r", backward, crossing->tlLinkIndex2);
                    // due GNEInternalLane aren't attribute carriers, we need to use the net grid
                    myViewNet->getNet()->getGrid().addAdditionalGLObject(internalLaneReverse);
                    myInternalLanes[crossing->tlLinkIndex2].push_back(internalLaneReverse);
                } else {
                    // draw only one lane for both directions
                    GNEInternalLane* internalLane = new GNEInternalLane(this, myTLSJunction->getCurrentJunction(), crossing->id, crossing->shape, crossing->tlLinkIndex);
                    // due GNEInternalLane aren't attribute carriers, we need to use the net grid
                    myViewNet->getNet()->getGrid().addAdditionalGLObject(internalLane);
                    myInternalLanes[crossing->tlLinkIndex].push_back(internalLane);
                }
            }
        }
    }
}


std::string
GNETLSEditorFrame::varDurString(SUMOTime dur) {
    return dur == NBTrafficLightDefinition::UNSPECIFIED_DURATION ? "   " : getSteps2Time(dur);
}


const std::vector<NBTrafficLightLogic::PhaseDefinition>&
GNETLSEditorFrame::getPhases() {
    return myEditedDef->getLogic()->getPhases();
}


void
GNETLSEditorFrame::handleChange(GNEInternalLane* lane) {
    myTLSModifications->setHaveModifications(true);
    if (myViewNet->changeAllPhases()) {
        const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = getPhases();
        for (int row = 0; row < (int)phases.size(); row++) {
            myEditedDef->getLogic()->setPhaseState(row, lane->getTLIndex(), lane->getLinkState());
        }
    } else {
        myEditedDef->getLogic()->setPhaseState(myTLSPhases->getPhaseTable()->getCurrentRow(), lane->getTLIndex(), lane->getLinkState());
    }
    myTLSPhases->initPhaseTable(myTLSPhases->getPhaseTable()->getCurrentRow());
    myTLSPhases->getPhaseTable()->setFocus();
}


void
GNETLSEditorFrame::handleMultiChange(GNELane* lane, FXObject* obj, FXSelector sel, void* eventData) {
    if (myEditedDef != nullptr) {
        myTLSModifications->setHaveModifications(true);
        const NBConnectionVector& links = myEditedDef->getControlledLinks();
        std::set<std::string> fromIDs;
        fromIDs.insert(lane->getMicrosimID());
        // if neither the lane nor its edge are selected, apply changes to the whole edge
        if (!lane->getParentEdge()->isAttributeCarrierSelected() && !lane->isAttributeCarrierSelected()) {
            for (auto it_lane : lane->getParentEdge()->getLanes()) {
                fromIDs.insert(it_lane->getMicrosimID());
            }
        } else {
            // if the edge is selected, apply changes to all lanes of all selected edges
            if (lane->getParentEdge()->isAttributeCarrierSelected()) {
                const auto selectedEdge = myViewNet->getNet()->getAttributeCarriers()->getSelectedEdges();
                for (const auto& edge : selectedEdge) {
                    for (auto it_lane : edge->getLanes()) {
                        fromIDs.insert(it_lane->getMicrosimID());
                    }
                }
            }
            // if the lane is selected, apply changes to all selected lanes
            if (lane->isAttributeCarrierSelected()) {
                const auto selectedLanes = myViewNet->getNet()->getAttributeCarriers()->getSelectedLanes();
                for (auto it_lane : selectedLanes) {
                    fromIDs.insert(it_lane->getMicrosimID());
                }
            }

        }
        // set new state for all connections from the chosen lane IDs
        for (auto it : links) {
            if (fromIDs.count(it.getFrom()->getLaneID(it.getFromLane())) > 0) {
                std::vector<GNEInternalLane*> lanes = myInternalLanes[it.getTLIndex()];
                for (auto it_lane : lanes) {
                    it_lane->onDefault(obj, sel, eventData);
                }
            }
        }
    }
}


bool
GNETLSEditorFrame::controlsEdge(GNEEdge* edge) const {
    if (myEditedDef != nullptr) {
        const NBConnectionVector& links = myEditedDef->getControlledLinks();
        for (auto it : links) {
            if (it.getFrom()->getID() == edge->getMicrosimID()) {
                return true;
            }
        }
    }
    return false;
}


void
GNETLSEditorFrame::editJunction(GNEJunction* junction) {
    if ((myTLSJunction->getCurrentJunction() == nullptr) || (!myTLSModifications->checkHaveModifications() && (junction != myTLSJunction->getCurrentJunction()))) {
        onCmdCancel(nullptr, 0, nullptr);
        myViewNet->getUndoList()->begin(GUIIcon::MODETLS, "modifying traffic light definition");
        myTLSJunction->setCurrentJunction(junction);
        myTLSAttributes->initTLSAttributes(myTLSJunction->getCurrentJunction());
        myTLSJunction->updateJunctionDescription();
        // only select TLS if getCurrentJunction exist
        if (myTLSJunction->getCurrentJunction()) {
            myTLSJunction->getCurrentJunction()->selectTLS(true);
        }
        if (myTLSAttributes->getNumberOfTLSDefinitions() > 0) {
            for (NBNode* node : myTLSAttributes->getCurrentTLSDefinition()->getNodes()) {
                myViewNet->getNet()->getAttributeCarriers()->retrieveJunction(node->getID())->selectTLS(true);
            }
        }
    } else {
        myViewNet->setStatusBarText("Unsaved modifications. Abort or Save");
    }
}


SUMOTime
GNETLSEditorFrame::getSUMOTime(const std::string& string) {
    return TIME2STEPS(GNEAttributeCarrier::parse<double>(string));
}

const std::string 
GNETLSEditorFrame::getSteps2Time(const SUMOTime value) {
    return toString(STEPS2TIME(value));
}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSAttributes - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSAttributes::TLSAttributes(GNETLSEditorFrame* TLSEditorParent) :
    FXGroupBoxModule(TLSEditorParent, "Traffic light Attributes"),
    myTLSEditorParent(TLSEditorParent) {

    // create frame, label and textfield for name (By default disabled)
    FXHorizontalFrame* typeFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    new FXLabel(typeFrame, toString(SUMO_ATTR_TYPE).c_str(), nullptr, GUIDesignLabelAttribute);
    myTLSType = new FXTextField(typeFrame, GUIDesignTextFieldNCol, myTLSEditorParent, 0, GUIDesignTextField);
    myTLSType->disable();

    // create frame, label and textfield for name (By default disabled)
    FXHorizontalFrame* idFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    new FXLabel(idFrame, toString(SUMO_ATTR_ID).c_str(), nullptr, GUIDesignLabelAttribute);
    myIDTextField = new FXTextField(idFrame, GUIDesignTextFieldNCol, myTLSEditorParent, MID_GNE_TLSFRAME_SWITCH, GUIDesignTextField);
    myIDTextField->disable();

    // create frame, label and comboBox for Program (By default hidden)
    FXHorizontalFrame* programFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    new FXLabel(programFrame, "program", nullptr, GUIDesignLabelAttribute);
    myProgramComboBox = new FXComboBox(programFrame, GUIDesignComboBoxNCol, myTLSEditorParent, MID_GNE_TLSFRAME_SWITCH, GUIDesignComboBoxAttribute);
    myProgramComboBox->disable();

    // create frame, label and TextField for Offset (By default disabled)
    FXHorizontalFrame* offsetFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    new FXLabel(offsetFrame, toString(SUMO_ATTR_OFFSET).c_str(), nullptr, GUIDesignLabelAttribute);
    myOffsetTextField = new FXTextField(offsetFrame, GUIDesignTextFieldNCol, myTLSEditorParent, MID_GNE_TLSFRAME_OFFSET, GUIDesignTextField);
    myOffsetTextField->disable();

    // create frame, label and TextField for Offset (By default disabled)
    FXHorizontalFrame* parametersFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    myButtonEditParameters = new FXButton(parametersFrame, "parameters", nullptr, myTLSEditorParent, MID_GNE_OPEN_PARAMETERS_DIALOG, GUIDesignButtonAttribute);
    myParametersTextField = new FXTextField(parametersFrame, GUIDesignTextFieldNCol, myTLSEditorParent, MID_GNE_TLSFRAME_PARAMETERS, GUIDesignTextField);
    myButtonEditParameters->disable();
    myParametersTextField->disable();
}


GNETLSEditorFrame::TLSAttributes::~TLSAttributes() {}


void
GNETLSEditorFrame::TLSAttributes::initTLSAttributes(GNEJunction* junction) {
    // clear definitions
    myTLSDefinitions.clear();
    // set TLS type
    myTLSType->setText(junction->getAttribute(SUMO_ATTR_TLTYPE).c_str());
    // enable id TextField
    myIDTextField->enable();
    // enable Offset
    myOffsetTextField->enable();
    myOffsetTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
    // enable parameters
    myButtonEditParameters->enable();
    myParametersTextField->enable();
    myParametersTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
    // obtain TLSs
    for (const auto &TLS : junction->getNBNode()->getControllingTLS()) {
        myTLSDefinitions.push_back(TLS);
        myIDTextField->setText(TLS->getID().c_str());
        myIDTextField->enable();
        myProgramComboBox->appendItem(TLS->getProgramID().c_str());
    }
    if (myTLSDefinitions.size() > 0) {
        myProgramComboBox->enable();
        myProgramComboBox->setCurrentItem(0);
        myProgramComboBox->setNumVisible(myProgramComboBox->getNumItems());
        myTLSEditorParent->onCmdDefSwitch(nullptr, 0, nullptr);
    }
}


void
GNETLSEditorFrame::TLSAttributes::clearTLSAttributes() {
    // clear definitions
    myTLSDefinitions.clear();
    // clear TLS type
    myTLSType->setText("");
    // clear and disable name TextField
    myIDTextField->setText("");
    myIDTextField->disable();
    // clear and disable myProgramComboBox
    myProgramComboBox->clearItems();
    myProgramComboBox->disable();
    // clear and disable Offset TextField
    myOffsetTextField->setText("");
    myOffsetTextField->disable();
    myOffsetTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
    // clear and disable parameters TextField
    myButtonEditParameters->disable();
    myParametersTextField->setText("");
    myParametersTextField->disable();
    myParametersTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
}


NBTrafficLightDefinition*
GNETLSEditorFrame::TLSAttributes::getCurrentTLSDefinition() const {
    return myTLSDefinitions.at(myProgramComboBox->getCurrentItem());
}


const std::string
GNETLSEditorFrame::TLSAttributes::getCurrentTLSProgramID() const {
    if (myProgramComboBox->getNumItems() == 0) {
        return "";
    } else {
        return myProgramComboBox->getText().text();
    }
}


int
GNETLSEditorFrame::TLSAttributes::getNumberOfTLSDefinitions() const {
    return (int)myTLSDefinitions.size();
}


int
GNETLSEditorFrame::TLSAttributes::getNumberOfPrograms() const {
    return myProgramComboBox->getNumItems();
}


SUMOTime
GNETLSEditorFrame::TLSAttributes::getOffset() const {
    return getSUMOTime(myOffsetTextField->getText().text());
}


void
GNETLSEditorFrame::TLSAttributes::setOffset(const SUMOTime& offset) {
    myOffsetTextField->setText(getSteps2Time(offset).c_str());
    myOffsetTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
}


bool
GNETLSEditorFrame::TLSAttributes::isValidOffset() {
    if (GNEAttributeCarrier::canParse<SUMOTime>(myOffsetTextField->getText().text())) {
        myOffsetTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
        return true;
    } else {
        myOffsetTextField->setTextColor(MFXUtils::getFXColor(RGBColor::RED));
        return false;
    }
}


std::string
GNETLSEditorFrame::TLSAttributes::getParameters() const {
    return myParametersTextField->getText().text();
}


void
GNETLSEditorFrame::TLSAttributes::setParameters(const std::string& parameters) {
    myParametersTextField->setText(parameters.c_str());
    myParametersTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
}


bool
GNETLSEditorFrame::TLSAttributes::isValidParameters() {
    if (Parameterised::areParametersValid(myParametersTextField->getText().text())) {
        myParametersTextField->setTextColor(MFXUtils::getFXColor(RGBColor::BLACK));
        return true;
    } else {
        myParametersTextField->setTextColor(MFXUtils::getFXColor(RGBColor::RED));
        return false;
    }
}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSJunction - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSJunction::TLSJunction(GNETLSEditorFrame* tlsEditorParent) :
    FXGroupBoxModule(tlsEditorParent, "Junction"),
    myCurrentJunction(nullptr) {
    // Create frame for junction ID
    FXHorizontalFrame* junctionIDFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    myLabelJunctionID = new FXLabel(junctionIDFrame, "Junction ID", nullptr, GUIDesignLabelAttribute);
    myTextFieldJunctionID = new FXTextField(junctionIDFrame, GUIDesignTextFieldNCol, this, MID_GNE_TLSFRAME_SELECT_JUNCTION, GUIDesignTextField);
    myTextFieldJunctionID->setEditable(false);
    // update junction description after creation
    updateJunctionDescription();
    // show TLS Junction
    show();
}


GNETLSEditorFrame::TLSJunction::~TLSJunction() {}


GNEJunction*
GNETLSEditorFrame::TLSJunction::getCurrentJunction() const {
    return myCurrentJunction;
}


void
GNETLSEditorFrame::TLSJunction::setCurrentJunction(GNEJunction* junction) {
    myCurrentJunction = junction;
}


void
GNETLSEditorFrame::TLSJunction::updateJunctionDescription() const {
    if (myCurrentJunction == nullptr) {
        myTextFieldJunctionID->setText("");
    } else {
        NBNode* nbn = myCurrentJunction->getNBNode();
        myTextFieldJunctionID->setText(nbn->getID().c_str());
    }
}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSDefinition - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSDefinition::TLSDefinition(GNETLSEditorFrame* TLSEditorParent) :
    FXGroupBoxModule(TLSEditorParent, "Traffic Light Programs") {
    // create auxiliar frames
    FXHorizontalFrame* horizontalFrameAux = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrameUniform);
    FXVerticalFrame * verticalFrameAuxA = new FXVerticalFrame(horizontalFrameAux, GUIDesignAuxiliarHorizontalFrame);
    FXVerticalFrame * verticalFrameAuxB = new FXVerticalFrame(horizontalFrameAux, GUIDesignAuxiliarHorizontalFrame);
    // create create tlDef button
    myNewTLProgram = new FXButton(verticalFrameAuxA, "Create\t\tCreate a new traffic light program",
                                  GUIIconSubSys::getIcon(GUIIcon::MODETLS), TLSEditorParent, MID_GNE_TLSFRAME_CREATE, GUIDesignButton);
    // create delete tlDef button
    myDeleteTLProgram = new FXButton(verticalFrameAuxB, "Delete\t\tDelete a traffic light program. If all programs are deleted the junction turns into a priority junction.",
                                     GUIIconSubSys::getIcon(GUIIcon::REMOVE), TLSEditorParent, MID_GNE_TLSFRAME_DELETE, GUIDesignButton);
    // create regenerate tlDef button
    myRegenerateTLProgram = new FXButton(verticalFrameAuxA, "Reset\t\tRegenerate TLS Program.",
                                         GUIIconSubSys::getIcon(GUIIcon::RELOAD), TLSEditorParent, MID_GNE_TLSFRAME_REGENERATE, GUIDesignButton);
    // show TLS TLSDefinition
    show();
}


GNETLSEditorFrame::TLSDefinition::~TLSDefinition() {}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSPhases - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSPhases::TLSPhases(GNETLSEditorFrame* TLSEditorParent) :
    FXGroupBoxModule(TLSEditorParent, "Phases", FXGroupBoxModule::Options::COLLAPSIBLE | FXGroupBoxModule::Options::EXTENSIBLE),
    myTLSEditorParent(TLSEditorParent),
    myTableFont(new FXFont(getApp(), "Courier New", 9)) {

    // create and configure phase table
    myTableScroll = new FXScrollWindow(getCollapsableFrame(), LAYOUT_FILL_X | LAYOUT_FIX_HEIGHT);
    myPhaseTable = new FXTable(myTableScroll, myTLSEditorParent, MID_GNE_TLSFRAME_PHASE_TABLE, GUIDesignTableLimitedHeight);
    myPhaseTable->setColumnHeaderMode(LAYOUT_FIX_HEIGHT);
    myPhaseTable->setColumnHeaderHeight(getApp()->getNormalFont()->getFontHeight() + getApp()->getNormalFont()->getFontAscent() / 2);
    myPhaseTable->setRowHeaderMode(LAYOUT_FIX_WIDTH);
    myPhaseTable->setRowHeaderWidth(0);
    myPhaseTable->hide();
    myPhaseTable->setFont(myTableFont);
    myPhaseTable->setHelpText("phase duration in seconds | phase state");

    // create total duration info label
    myCycleDuration = new FXLabel(getCollapsableFrame(), "", nullptr, GUIDesignLabelLeft);

    // using FXMatrix for tabular button layout would have been cleaner but the
    // below attempt did not make the buttons fill available horizontal space
    // FXMatrix* phaseButtons = new FXMatrix(getCollapsableFrame(), 2, LAYOUT_FILL_X | MATRIX_BY_COLUMNS);

    FXHorizontalFrame* phaseButtons = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrameUniform);
    FXVerticalFrame* col1 = new FXVerticalFrame(phaseButtons, GUIDesignAuxiliarHorizontalFrame); // left button columm
    FXVerticalFrame* col2 = new FXVerticalFrame(phaseButtons, GUIDesignAuxiliarHorizontalFrame); // right button column

    // create new phase button
    myInsertDuplicateButton = new FXButton(col1, "Insert Phase\t\tInsert new phase after the selected phase. The new state is deduced from the selected phase.", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_PHASE_CREATE, GUIDesignButton);
    // create delete phase button
    myDeleteSelectedPhaseButton = new FXButton(col2, "Delete Phase\t\tDelete selected phase", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_PHASE_DELETE, GUIDesignButton);

    // create cleanup states button
    new FXButton(col1, "Clean States\t\tClean unused states from all phase. (Not allowed for multiple programs)", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_CLEANUP, GUIDesignButton);

    // add unused states button
    new FXButton(col2, "Add States\t\tExtend the state vector for all phases by one entry (unused until a connection or crossing is assigned to the new index).", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_ADDUNUSED, GUIDesignButton);

    // group states button
    new FXButton(col1, "Group Signals\t\tShorten state definition by letting connections with the same signal states use the same index. (Not allowed for multiple programs)", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_GROUP_STATES, GUIDesignButton);

    // ungroup states button
    new FXButton(col2, "Ungroup Signals\t\tLet every connection use a distinct index (reverse state grouping). (Not allowed for multiple programs)", nullptr, myTLSEditorParent, MID_GNE_TLSFRAME_UNGROUP_STATES, GUIDesignButton);
    // show TLSFile
    show();
}


GNETLSEditorFrame::TLSPhases::~TLSPhases() {
    delete myTableFont;
}


FXTable*
GNETLSEditorFrame::TLSPhases::getPhaseTable() const {
    return myPhaseTable;
}


void
GNETLSEditorFrame::TLSPhases::initPhaseTable(int index) {
    myPhaseTable->setVisibleRows(1);
    myPhaseTable->setVisibleColumns(2);
    myPhaseTable->hide();
    if (myTLSEditorParent->myTLSAttributes->getNumberOfTLSDefinitions() > 0) {
        if (myTLSEditorParent->myEditedDef->getType() == TrafficLightType::STATIC) {
            initStaticPhaseTable(index);
        } else if (myTLSEditorParent->myEditedDef->getType() == TrafficLightType::ACTUATED) {
            initActuatedPhaseTable(index);
        } else if(myTLSEditorParent->myEditedDef->getType() == TrafficLightType::DELAYBASED) {
            initDelayBasePhaseTable(index);
        } else if (myTLSEditorParent->myEditedDef->getType() == TrafficLightType::NEMA) {
            initNEMAPhaseTable(index);
        }
    }
    update();
}


void
GNETLSEditorFrame::TLSPhases::showCycleDuration() {
    myCycleDuration->show();
}


void
GNETLSEditorFrame::TLSPhases::hideCycleDuration() {
    myCycleDuration->hide();
}

void
GNETLSEditorFrame::TLSPhases::updateCycleDuration() {
    SUMOTime cycleDuration = 0;
    for (auto it : myTLSEditorParent->getPhases()) {
        cycleDuration += it.duration;
    }
    std::string text = "Cycle time: " + getSteps2Time(cycleDuration);
    myCycleDuration->setText(text.c_str());
}


void 
GNETLSEditorFrame::TLSPhases::initStaticPhaseTable(const int index) {
    // declare constants for columns
    const int cols = 4;
    const int colDuration = 0;
    const int colState = 1;
    const int colNext = 2;
    const int colName = 3;
    // get phases
    const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = myTLSEditorParent->getPhases();
    // adjust table
    myPhaseTable->setTableSize((int)phases.size(), cols);
    myPhaseTable->setVisibleRows((int)phases.size());
    myPhaseTable->setVisibleColumns(cols);
    // fill rows
    for (int row = 0; row < (int)phases.size(); row++) {
        myPhaseTable->setItemText(row, colDuration, getSteps2Time(phases[row].duration).c_str());
        myPhaseTable->setItemText(row, colState, phases[row].state.c_str());
        myPhaseTable->setItemText(row, colNext, phases[row].next.size() > 0 ? toString(phases[row].next).c_str() : " ");
        myPhaseTable->setItemText(row, colName, phases[row].name.c_str());
        myPhaseTable->getItem(row, 1)->setJustify(FXTableItem::LEFT);
    }
    // set columns
    myPhaseTable->fitColumnsToContents(0, cols);
    myPhaseTable->setColumnText(colDuration, "dur");
    myPhaseTable->setColumnText(colState, "state");
    myPhaseTable->setColumnText(colNext, "nxt");
    myPhaseTable->setColumnText(colName, "name");
    myPhaseTable->setColumnWidth(colNext, MAX2(myPhaseTable->getColumnWidth(colNext), 30));
    myPhaseTable->setColumnWidth(colName, MAX2(myPhaseTable->getColumnWidth(colName), 45));
    // set rows
    myPhaseTable->setHeight((int)phases.size() * 21 + 21); // experimental
    myPhaseTable->setCurrentItem(index, 0);
    myPhaseTable->selectRow(index, true);
    myPhaseTable->show();
    myPhaseTable->setFocus();
    myTableScroll->setHeight(myPhaseTable->getHeight() + 15);
    // neither my myPhaseTable->getWidth nor getDefaultWidth return the sum of column widths
    // however, the scroll pane uses getDefaultWidth to determine the
    // horizontal scrolling area which can only be changed via
    // getDefColumnWidth, hence the baroque work-around
    int neededWidth = 0;
    for (int i = 0; i < cols; i++) {
        neededWidth += myPhaseTable->getColumnWidth(i);
    }
    myPhaseTable->setDefColumnWidth(neededWidth / cols);
}


void 
GNETLSEditorFrame::TLSPhases::initActuatedPhaseTable(const int index) {
    // declare constants for columns
    const int cols = 8;
    const int colDuration = 0;
    const int colMinDur = 1;
    const int colMaxDur = 2;
    const int colState = 3;
    const int colEarliestEnd = 4;
    const int colLatestEnd = 5;
    const int colNext = 6;
    const int colName = 7;
    // get phases
    const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = myTLSEditorParent->getPhases();
    // adjust table
    myPhaseTable->setTableSize((int)phases.size(), cols);
    myPhaseTable->setVisibleRows((int)phases.size());
    myPhaseTable->setVisibleColumns(cols);
    // fill rows
    for (int row = 0; row < (int)phases.size(); row++) {
        myPhaseTable->setItemText(row, colDuration, getSteps2Time(phases[row].duration).c_str());
        myPhaseTable->setItemText(row, colMinDur, varDurString(phases[row].minDur).c_str());
        myPhaseTable->setItemText(row, colMaxDur, varDurString(phases[row].maxDur).c_str());
        myPhaseTable->setItemText(row, colState, phases[row].state.c_str());
        myPhaseTable->setItemText(row, colEarliestEnd, varDurString(phases[row].earliestEnd).c_str());
        myPhaseTable->setItemText(row, colLatestEnd, varDurString(phases[row].latestEnd).c_str());
        myPhaseTable->setItemText(row, colNext, phases[row].next.size() > 0 ? toString(phases[row].next).c_str() : " ");
        myPhaseTable->setItemText(row, colName, phases[row].name.c_str());
        myPhaseTable->getItem(row, 1)->setJustify(FXTableItem::LEFT);
    }
    // set columns
    myPhaseTable->fitColumnsToContents(0, cols);
    myPhaseTable->setColumnText(colDuration, "dur");
    myPhaseTable->setColumnWidth(colDuration, MAX2(myPhaseTable->getColumnWidth(colDuration), 35));
    myPhaseTable->setColumnText(colMinDur, "min");
    myPhaseTable->setColumnWidth(colMinDur, MAX2(myPhaseTable->getColumnWidth(colMinDur), 35));
    myPhaseTable->setColumnText(colMaxDur, "max");
    myPhaseTable->setColumnWidth(colMaxDur, MAX2(myPhaseTable->getColumnWidth(colMaxDur), 35));
    myPhaseTable->setColumnText(colEarliestEnd, "ear.end");
    myPhaseTable->setColumnWidth(colEarliestEnd, MAX2(myPhaseTable->getColumnWidth(colEarliestEnd), 35));
    myPhaseTable->setColumnText(colLatestEnd, "lat.end");
    myPhaseTable->setColumnWidth(colLatestEnd, MAX2(myPhaseTable->getColumnWidth(colLatestEnd), 35));
    myPhaseTable->setColumnText(colState, "state");
    myPhaseTable->setColumnWidth(colState, MAX2(myPhaseTable->getColumnWidth(colState), 30));
    myPhaseTable->setColumnText(colNext, "nxt");
    myPhaseTable->setColumnWidth(colNext, MAX2(myPhaseTable->getColumnWidth(colNext), 30));
    myPhaseTable->setColumnText(colName, "name");
    myPhaseTable->setColumnWidth(colName, MAX2(myPhaseTable->getColumnWidth(colName), 45));
    // set rows
    myPhaseTable->setHeight((int)phases.size() * 21 + 21); // experimental
    myPhaseTable->setCurrentItem(index, 0);
    myPhaseTable->selectRow(index, true);
    myPhaseTable->show();
    myPhaseTable->setFocus();
    myTableScroll->setHeight(myPhaseTable->getHeight() + 15);
    // neither my myPhaseTable->getWidth nor getDefaultWidth return the sum of column widths
    // however, the scroll pane uses getDefaultWidth to determine the
    // horizontal scrolling area which can only be changed via
    // getDefColumnWidth, hence the baroque work-around
    int neededWidth = 0;
    for (int i = 0; i < cols; i++) {
        neededWidth += myPhaseTable->getColumnWidth(i);
    }
    myPhaseTable->setDefColumnWidth(neededWidth / cols);
}


void 
GNETLSEditorFrame::TLSPhases::initDelayBasePhaseTable(const int index) {
    // declare constants for columns
    const int cols = 8;
    const int colDuration = 0;
    const int colMinDur = 1;
    const int colMaxDur = 2;
    const int colState = 3;
    const int colNext = 4;
    const int colName = 5;
    // get phases
    const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = myTLSEditorParent->getPhases();
    // adjust table
    myPhaseTable->setTableSize((int)phases.size(), cols);
    myPhaseTable->setVisibleRows((int)phases.size());
    myPhaseTable->setVisibleColumns(cols);
    // fill rows
    for (int row = 0; row < (int)phases.size(); row++) {
        myPhaseTable->setItemText(row, colDuration, getSteps2Time(phases[row].duration).c_str());
        myPhaseTable->setItemText(row, colMinDur, varDurString(phases[row].minDur).c_str());
        myPhaseTable->setItemText(row, colMaxDur, varDurString(phases[row].maxDur).c_str());
        myPhaseTable->setItemText(row, colState, phases[row].state.c_str());
        myPhaseTable->setItemText(row, colNext, phases[row].next.size() > 0 ? toString(phases[row].next).c_str() : " ");
        myPhaseTable->setItemText(row, colName, phases[row].name.c_str());
        myPhaseTable->getItem(row, 1)->setJustify(FXTableItem::LEFT);
    }
    // set columns
    myPhaseTable->fitColumnsToContents(0, cols);
    myPhaseTable->setColumnText(colDuration, "dur");
    myPhaseTable->setColumnWidth(colDuration, MAX2(myPhaseTable->getColumnWidth(colDuration), 35));
    myPhaseTable->setColumnText(colMinDur, "min");
    myPhaseTable->setColumnWidth(colMinDur, MAX2(myPhaseTable->getColumnWidth(colMinDur), 35));
    myPhaseTable->setColumnText(colMaxDur, "max");
    myPhaseTable->setColumnWidth(colMaxDur, MAX2(myPhaseTable->getColumnWidth(colMaxDur), 35));
    myPhaseTable->setColumnText(colState, "state");
    myPhaseTable->setColumnWidth(colState, MAX2(myPhaseTable->getColumnWidth(colState), 30));
    myPhaseTable->setColumnText(colNext, "nxt");
    myPhaseTable->setColumnWidth(colNext, MAX2(myPhaseTable->getColumnWidth(colNext), 30));
    myPhaseTable->setColumnText(colName, "name");
    myPhaseTable->setColumnWidth(colName, MAX2(myPhaseTable->getColumnWidth(colName), 45));
    // set rows
    myPhaseTable->setHeight((int)phases.size() * 21 + 21); // experimental
    myPhaseTable->setCurrentItem(index, 0);
    myPhaseTable->selectRow(index, true);
    myPhaseTable->show();
    myPhaseTable->setFocus();
    myTableScroll->setHeight(myPhaseTable->getHeight() + 15);
    // neither my myPhaseTable->getWidth nor getDefaultWidth return the sum of column widths
    // however, the scroll pane uses getDefaultWidth to determine the
    // horizontal scrolling area which can only be changed via
    // getDefColumnWidth, hence the baroque work-around
    int neededWidth = 0;
    for (int i = 0; i < cols; i++) {
        neededWidth += myPhaseTable->getColumnWidth(i);
    }
    myPhaseTable->setDefColumnWidth(neededWidth / cols);
}


void 
GNETLSEditorFrame::TLSPhases::initNEMAPhaseTable(const int index) {
    // declare constants for columns
    const int cols = 9;
    const int colDuration = 0;
    const int colMinDur = 1;
    const int colMaxDur = 2;
    const int colState = 3;
    const int colVehExt = 4;
    const int colYellow = 5;
    const int colRed = 6;
    const int colNext = 7;
    const int colName = 8;
    // get phases
    const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = myTLSEditorParent->getPhases();
    // adjust table
    myPhaseTable->setTableSize((int)phases.size(), cols);
    myPhaseTable->setVisibleRows((int)phases.size());
    myPhaseTable->setVisibleColumns(cols);
    // fill rows
    for (int row = 0; row < (int)phases.size(); row++) {
        myPhaseTable->setItemText(row, colDuration, getSteps2Time(phases[row].duration).c_str());
        myPhaseTable->setItemText(row, colMinDur, varDurString(phases[row].minDur).c_str());
        myPhaseTable->setItemText(row, colMaxDur, varDurString(phases[row].maxDur).c_str());
        myPhaseTable->setItemText(row, colState, phases[row].state.c_str());
        myPhaseTable->setItemText(row, colVehExt, varDurString(phases[row].vehExt).c_str());
        myPhaseTable->setItemText(row, colYellow, varDurString(phases[row].yellow).c_str());
        myPhaseTable->setItemText(row, colRed, varDurString(phases[row].red).c_str());
        myPhaseTable->setItemText(row, colNext, phases[row].next.size() > 0 ? toString(phases[row].next).c_str() : " ");
        myPhaseTable->setItemText(row, colName, phases[row].name.c_str());
        myPhaseTable->getItem(row, 1)->setJustify(FXTableItem::LEFT);
    }
    // set columns
    myPhaseTable->fitColumnsToContents(0, cols);
    myPhaseTable->setColumnText(colDuration, "dur");
    myPhaseTable->setColumnWidth(colDuration, MAX2(myPhaseTable->getColumnWidth(colDuration), 35));
    myPhaseTable->setColumnText(colMinDur, "min");
    myPhaseTable->setColumnWidth(colMinDur, MAX2(myPhaseTable->getColumnWidth(colMinDur), 35));
    myPhaseTable->setColumnText(colMaxDur, "max");
    myPhaseTable->setColumnWidth(colMaxDur, MAX2(myPhaseTable->getColumnWidth(colMaxDur), 35));
    myPhaseTable->setColumnText(colState, "state");
    myPhaseTable->setColumnWidth(colState, MAX2(myPhaseTable->getColumnWidth(colState), 30));
    myPhaseTable->setColumnText(colVehExt, "v.ext");
    myPhaseTable->setColumnWidth(colVehExt, MAX2(myPhaseTable->getColumnWidth(colVehExt), 45));
    myPhaseTable->setColumnText(colYellow, "yel");
    myPhaseTable->setColumnWidth(colYellow, MAX2(myPhaseTable->getColumnWidth(colYellow), 35));
    myPhaseTable->setColumnText(colRed, "red");
    myPhaseTable->setColumnWidth(colRed, MAX2(myPhaseTable->getColumnWidth(colRed), 35));
    myPhaseTable->setColumnText(colNext, "nxt");
    myPhaseTable->setColumnWidth(colNext, MAX2(myPhaseTable->getColumnWidth(colNext), 30));
    myPhaseTable->setColumnText(colName, "name");
    myPhaseTable->setColumnWidth(colName, MAX2(myPhaseTable->getColumnWidth(colName), 45));
    // set rows
    myPhaseTable->setHeight((int)phases.size() * 21 + 21); // experimental
    myPhaseTable->setCurrentItem(index, 0);
    myPhaseTable->selectRow(index, true);
    myPhaseTable->show();
    myPhaseTable->setFocus();
    myTableScroll->setHeight(myPhaseTable->getHeight() + 15);
    // neither my myPhaseTable->getWidth nor getDefaultWidth return the sum of column widths
    // however, the scroll pane uses getDefaultWidth to determine the
    // horizontal scrolling area which can only be changed via
    // getDefColumnWidth, hence the baroque work-around
    int neededWidth = 0;
    for (int i = 0; i < cols; i++) {
        neededWidth += myPhaseTable->getColumnWidth(i);
    }
    myPhaseTable->setDefColumnWidth(neededWidth / cols);
}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSModifications - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSModifications::TLSModifications(GNETLSEditorFrame* TLSEditorParent) :
    FXGroupBoxModule(TLSEditorParent, "Modifications"),
    myTLSEditorParent(TLSEditorParent),
    myHaveModifications(false) {
    FXHorizontalFrame* buttonsFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    // create save modifications button
    mySaveModificationsButtons = new FXButton(buttonsFrame, "Save\t\tSave program modifications (Enter)",
            GUIIconSubSys::getIcon(GUIIcon::OK), myTLSEditorParent, MID_OK, GUIDesignButton);
    // create discard modifications buttons
    myDiscardModificationsButtons = new FXButton(buttonsFrame, "Cancel\t\tDiscard program modifications (Esc)",
            GUIIconSubSys::getIcon(GUIIcon::CANCEL), myTLSEditorParent, MID_CANCEL, GUIDesignButton);
    // show TLSModifications
    show();
}


GNETLSEditorFrame::TLSModifications::~TLSModifications() {}


bool
GNETLSEditorFrame::TLSModifications::checkHaveModifications() const {
    return myHaveModifications;
}


void
GNETLSEditorFrame::TLSModifications::setHaveModifications(bool value) {
    myHaveModifications = value;
}

// ---------------------------------------------------------------------------
// GNETLSEditorFrame::TLSFile - methods
// ---------------------------------------------------------------------------

GNETLSEditorFrame::TLSFile::TLSFile(GNETLSEditorFrame* TLSEditorParent) :
    FXGroupBoxModule(TLSEditorParent, "TLS Program File"),
    myTLSEditorParent(TLSEditorParent) {
    FXHorizontalFrame* buttonsFrame = new FXHorizontalFrame(getCollapsableFrame(), GUIDesignAuxiliarHorizontalFrame);
    // create create tlDef button
    myLoadTLSProgramButton = new FXButton(buttonsFrame, "Load\t\tLoad TLS program from additional file", GUIIconSubSys::getIcon(GUIIcon::OPEN_CONFIG), this, MID_GNE_TLSFRAME_LOAD_PROGRAM, GUIDesignButton);
    // create create tlDef button
    mySaveTLSProgramButton = new FXButton(buttonsFrame, "Save\t\tSave TLS program to additional file", GUIIconSubSys::getIcon(GUIIcon::SAVE), this, MID_GNE_TLSFRAME_SAVE_PROGRAM, GUIDesignButton);
    // show TLSFile
    show();
}


GNETLSEditorFrame::TLSFile::~TLSFile() {}


long
GNETLSEditorFrame::TLSFile::onCmdLoadTLSProgram(FXObject*, FXSelector, void*) {
    FXFileDialog opendialog(getCollapsableFrame(), "Load TLS Program");
    opendialog.setIcon(GUIIconSubSys::getIcon(GUIIcon::MODETLS));
    opendialog.setSelectMode(SELECTFILE_EXISTING);
    opendialog.setPatternList("XML files (*.xml)\nAll files (*)");
    if (gCurrentFolder.length() != 0) {
        opendialog.setDirectory(gCurrentFolder);
    }
    if (opendialog.execute()) {
        // run parser
        NBTrafficLightLogicCont tmpTLLCont;
        NIXMLTrafficLightsHandler tllHandler(tmpTLLCont, myTLSEditorParent->myViewNet->getNet()->getEdgeCont(), true);
        tmpTLLCont.insert(myTLSEditorParent->myEditedDef);
        XMLSubSys::runParser(tllHandler, opendialog.getFilename().text());

        NBLoadedSUMOTLDef* newDefSameProgram = nullptr;
        std::set<NBLoadedSUMOTLDef*> newDefsOtherProgram;
        for (auto item : tmpTLLCont.getPrograms(myTLSEditorParent->myEditedDef->getID())) {
            if (item.second != myTLSEditorParent->myEditedDef) {
                NBLoadedSUMOTLDef* sdef = dynamic_cast<NBLoadedSUMOTLDef*>(item.second);
                if (item.first == myTLSEditorParent->myEditedDef->getProgramID()) {
                    newDefSameProgram = sdef;
                } else {
                    newDefsOtherProgram.insert(sdef);
                }
            }
        }
        const int newPrograms = (int)newDefsOtherProgram.size();
        if (newPrograms > 0 || newDefSameProgram != nullptr) {
            std::vector<NBNode*> nodes = myTLSEditorParent->myEditedDef->getNodes();
            for (auto newProg : newDefsOtherProgram) {
                for (auto it_node : nodes) {
                    GNEJunction* junction = myTLSEditorParent->getViewNet()->getNet()->getAttributeCarriers()->retrieveJunction(it_node->getID());
                    myTLSEditorParent->getViewNet()->getUndoList()->add(new GNEChange_TLS(junction, newProg, true), true);
                }
            }
            if (newPrograms > 0) {
                WRITE_MESSAGE("Loaded " + toString(newPrograms) + " new programs for tlLogic '" + myTLSEditorParent->myEditedDef->getID() + "'");
            }
            if (newDefSameProgram != nullptr) {
                // replace old program when loading the same program ID
                myTLSEditorParent->myEditedDef = newDefSameProgram;
                WRITE_MESSAGE("Updated program '" + newDefSameProgram->getProgramID() +  "' for tlLogic '" + myTLSEditorParent->myEditedDef->getID() + "'");
            }
        } else {
            if (tllHandler.getSeenIDs().count(myTLSEditorParent->myEditedDef->getID()) == 0)  {
                myTLSEditorParent->getViewNet()->setStatusBarText("No programs found for traffic light '" + myTLSEditorParent->myEditedDef->getID() + "'");
            }
        }

        // clean up temporary container to avoid deletion of defs when it's destruct is called
        for (NBTrafficLightDefinition* def : tmpTLLCont.getDefinitions()) {
            tmpTLLCont.removeProgram(def->getID(), def->getProgramID(), false);
        }

        myTLSEditorParent->myTLSPhases->initPhaseTable();
        myTLSEditorParent->myTLSModifications->setHaveModifications(true);
    }
    return 0;
}


long
GNETLSEditorFrame::TLSFile::onCmdSaveTLSProgram(FXObject*, FXSelector, void*) {
    FXString file = MFXUtils::getFilename2Write(this,
                    "Save TLS Program as", ".xml",
                    GUIIconSubSys::getIcon(GUIIcon::MODETLS),
                    gCurrentFolder);
    // check file
    if (file != "") {
        // add xml extension
        file = FileHelpers::addExtension(file.text(), ".xml").c_str();
        OutputDevice& device = OutputDevice::getDevice(file.text());
        // save program
        device.writeXMLHeader("additional", "additional_file.xsd");
        device.openTag(SUMO_TAG_TLLOGIC);
        device.writeAttr(SUMO_ATTR_ID, myTLSEditorParent->myEditedDef->getLogic()->getID());
        device.writeAttr(SUMO_ATTR_TYPE, myTLSEditorParent->myEditedDef->getLogic()->getType());
        device.writeAttr(SUMO_ATTR_PROGRAMID, myTLSEditorParent->myEditedDef->getLogic()->getProgramID());
        device.writeAttr(SUMO_ATTR_OFFSET, writeSUMOTime(myTLSEditorParent->myEditedDef->getLogic()->getOffset()));
        myTLSEditorParent->myEditedDef->writeParams(device);
        // write the phases
        const bool TLSActuated = (myTLSEditorParent->myEditedDef->getLogic()->getType() == TrafficLightType::ACTUATED);
        const bool TLSDelayBased = (myTLSEditorParent->myEditedDef->getLogic()->getType() == TrafficLightType::DELAYBASED);
        const bool TLSNEMA = (myTLSEditorParent->myEditedDef->getLogic()->getType() == TrafficLightType::NEMA);
        // write the phases
        const std::vector<NBTrafficLightLogic::PhaseDefinition>& phases = myTLSEditorParent->myEditedDef->getLogic()->getPhases();
        for (const auto &phase : phases) {
            device.openTag(SUMO_TAG_PHASE);
            device.writeAttr(SUMO_ATTR_DURATION, writeSUMOTime(phase.duration));
            device.writeAttr(SUMO_ATTR_STATE, phase.state);
            // write specific actuated parameters
            if (TLSActuated || TLSDelayBased) {
                if (phase.minDur != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MINDURATION, writeSUMOTime(phase.minDur));
                }
                if (phase.maxDur != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MAXDURATION, writeSUMOTime(phase.maxDur));
                }
                if (phase.earliestEnd != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MAXDURATION, writeSUMOTime(phase.maxDur));
                }
                if (phase.earliestEnd != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_EARLIEST_END, writeSUMOTime(phase.maxDur));
                }
                if (phase.latestEnd != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_LATEST_END, writeSUMOTime(phase.maxDur));
                }
            }
            // write specific NEMA parameters
            if (TLSNEMA) {
                if (phase.minDur != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MINDURATION, writeSUMOTime(phase.minDur));
                }
                if (phase.maxDur != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MAXDURATION, writeSUMOTime(phase.maxDur));
                }
                if (phase.vehExt != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MINDURATION, writeSUMOTime(phase.vehExt));
                }
                if (phase.red != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MAXDURATION, writeSUMOTime(phase.red));
                }
                if (phase.yellow != NBTrafficLightDefinition::UNSPECIFIED_DURATION) {
                    device.writeAttr(SUMO_ATTR_MAXDURATION, writeSUMOTime(phase.yellow));
                }
            }
            device.closeTag();
        }
        device.close();
    }
    return 1;
}


std::string
GNETLSEditorFrame::TLSFile::writeSUMOTime(SUMOTime steps) {
    const double time = STEPS2TIME(steps);
    if (time == std::floor(time)) {
        return toString(int(time));
    } else {
        return toString(time);
    }
}

long
GNETLSEditorFrame::TLSFile::onUpdNeedsDef(FXObject* o, FXSelector, void*) {
    const bool enable = myTLSEditorParent->myTLSAttributes->getNumberOfTLSDefinitions() > 0;
    o->handle(getCollapsableFrame(), FXSEL(SEL_COMMAND, enable ? FXWindow::ID_ENABLE : FXWindow::ID_DISABLE), nullptr);
    return 1;
}


/****************************************************************************/
