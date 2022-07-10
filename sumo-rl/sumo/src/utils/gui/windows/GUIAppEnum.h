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
/// @file    GUIAppEnum.h
/// @author  Daniel Krajzewicz
/// @author  Jakob Erdmann
/// @author  Michael Behrisch
/// @author  Andreas Gaubatz
/// @date    Mon, 08.03.2004
///
// Message and object IDs used by the FOX-version of the gui
/****************************************************************************/
#pragma once
#include <config.h>

#include <utils/foxtools/fxheader.h>


// ===========================================================================
// enumeration
// ===========================================================================
/**
 * @brief Message and object IDs used by the FOX-version of the gui
 */
enum {
    /// @brief Show about - dialog
    ID_ABOUT = FXMainWindow::ID_LAST,

    /// @name hotkeys
    /// @{

    /// @brief hotkey for mode editing additionals AND stops
    MID_HOTKEY_A_MODE_ADDITIONAL_STOP,
    /// @brief hotkey for mode connecting lanes ABD person plan
    MID_HOTKEY_C_MODE_CONNECT_PERSONPLAN,
    /// @brief hotkey for mode deleting things
    MID_HOTKEY_D_MODE_DELETE,
    /// @brief hotkey for mode adding edges AND edgeDatas
    MID_HOTKEY_E_MODE_EDGE_EDGEDATA,
    /// @brief hotkey for mode container
    MID_HOTKEY_G_MODE_CONTAINER,
    /// @brief hotkey for mode prohibition AND container plan
    MID_HOTKEY_H_MODE_PROHIBITION_CONTAINERPLAN,
    /// @brief hotkey for mode inspecting object attributes
    MID_HOTKEY_I_MODE_INSPECT,
    /// @brief hotkey for mode moving element
    MID_HOTKEY_M_MODE_MOVE,
    /// @brief hotkey for mode creating polygons
    MID_HOTKEY_P_MODE_POLYGON_PERSON,
    /// @brief hotkey for mode selecting objects
    MID_HOTKEY_S_MODE_SELECT,
    /// @brief hotkey for mode editing crossing, routes and edge rel datas
    MID_HOTKEY_R_MODE_CROSSING_ROUTE_EDGERELDATA,
    /// @brief hotkey for mode editing TLS AND Vehicle Types
    MID_HOTKEY_T_MODE_TLS_TYPE,
    /// @brief hotkey for mode create vehicles
    MID_HOTKEY_V_MODE_VEHICLE,
    /// @brief hotkey for mode editing overhead wires
    MID_HOTKEY_W_MODE_WIRE,
    /// @brief hotkey for mode editing TAZ and TAZRel
    MID_HOTKEY_Z_MODE_TAZ_TAZREL,

    /// @}

    /// @name Ctrl hotkeys
    /// @{

    /// @brief Start the simulation in SUMO and open Additionals Elements in NETEDIT
    MID_HOTKEY_CTRL_A_STARTSIMULATION_OPENADDITIONALS,
    /// @brief Edit simulation breakpoints in SUMO and open Data Elements in NETEDIT
    MID_HOTKEY_CTRL_B_EDITBREAKPOINT_OPENDATAELEMENTS,
    /// @brief Copy the current selection / element
    MID_HOTKEY_CTRL_C_COPY,
    /// @brief Perform a single simulation step in SUMO and open Demand Elements in NETEDIT
    MID_HOTKEY_CTRL_D_SINGLESIMULATIONSTEP_OPENDEMANDELEMENTS,
    /// @brief Toggle Gaming mode in SUMO and grid in NETEDIT
    MID_HOTKEY_CTRL_G_GAMINGMODE_TOGGLEGRID,
    /// @brief open edge type files
    MID_HOTKEY_CTRL_H_OPENEDGETYPES,
    /// @brief Open viewport editor
    MID_HOTKEY_CTRL_I_EDITVIEWPORT,
    /// @brief Fullscreen mode - menu entry
    MID_HOTKEY_CTRL_F_FULSCREENMODE,
    /// @brief toggle draw junction shape
    MID_HOTKEY_CTRL_J_TOOGLEDRAWJUNCTIONSHAPE,
    /// @brief Load file with TLS Programs
    MID_HOTKEY_CTRL_K_OPENTLSPROGRAMS,
    /// @brief save network as plain XML
    MID_HOTKEY_CTRL_L_SAVEASPLAINXML,
    /// @brief create new empty network
    MID_HOTKEY_CTRL_N_NEWNETWORK,
    /// @brief Main window closes
    MID_HOTKEY_CTRL_Q_CLOSE,
    /// @brief Stop the simulation in SUMO and save network in NETEDIT
    MID_HOTKEY_CTRL_S_STOPSIMULATION_SAVENETWORK,
    /// @brief Reload the previously loaded simulation
    MID_HOTKEY_CTRL_R_RELOAD,
    /// @brief Quick-Reload the previously loaded simulation (keep the net)
    MID_HOTKEY_CTRL_QUICK_RELOAD,
    /// @brief Open current network in SUMO or in NETEDIT
    MID_HOTKEY_CTRL_T_OPENSUMONETEDIT,
    /// @brief paste the current selection / element
    MID_HOTKEY_CTRL_V_PASTE,
    /// @brief Close simulation - ID
    MID_HOTKEY_CTRL_W_CLOSESIMULATION,
    /// @brief cut the current selection / element
    MID_HOTKEY_CTRL_X_CUT,
    /// @brief Undo
    MID_HOTKEY_CTRL_Y_REDO,
    /// @brief Redo
    MID_HOTKEY_CTRL_Z_UNDO,

    /// @}

    /// @name Alt hotkeys
    /// @{

    /// @brief toggle edit option
    MID_HOTKEY_ALT_0_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_1_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_2_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_3_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_4_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_5_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_6_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_7_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_8_TOGGLEEDITOPTION,
    /// @brief toggle edit option
    MID_HOTKEY_ALT_9_TOGGLEEDITOPTION,
    /// @brief Main window closes
    MID_HOTKEY_ALT_F4_CLOSE,

    /// @}

    /// @name Ctrl + Shift hotkeys
    /// @{

    /// @brief Save Additional Elements
    MID_HOTKEY_CTRL_SHIFT_A_SAVEADDITIONALS,
    /// @brief save Edge Types
    MID_HOTKEY_CTRL_SHIFT_H_SAVEEDGETYPES,
    /// @brief save Data Elements
    MID_HOTKEY_CTRL_SHIFT_B_SAVEDATAELEMENTS,
    /// @brief Save Demand Elements
    MID_HOTKEY_CTRL_SHIFT_D_SAVEDEMANDELEMENTS,
    /// @brief save TLS Programs
    MID_HOTKEY_CTRL_SHIFT_K_SAVETLS,
    /// @brief Open a New Window
    MID_HOTKEY_CTRL_SHIFT_N_NEWWINDOW,
    /// @brief save network as
    MID_HOTKEY_CTRL_SHIFT_S_SAVENETWORK_AS,
    /// @brief Force save network (flag)
    MID_HOTKEY_CTRL_SHIFT_T_FORCESAVENETEWORK,
    /// @brief Force save additionals (flag)
    MID_HOTKEY_CTRL_SHIFT_U_FORCESAVEADDITIONALS,
    /// @brief Force save demand elements (flag)
    MID_HOTKEY_CTRL_SHIFT_V_FORCESAVEDEMANDELEMENTS,
    /// @brief Force save data elements (flag)
    MID_HOTKEY_CTRL_SHIFT_W_FORCESAVEDATAELEMENTS,

    /// @}

    /// @name Functional hotkeys
    /// @{

    /// @brief open online documentation
    MID_HOTKEY_F1_ONLINEDOCUMENTATION,
    /// @brief select network supermode in NETEDIT
    MID_HOTKEY_F2_SUPERMODE_NETWORK,
    /// @brief select demand supermode in NETEDIT
    MID_HOTKEY_F3_SUPERMODE_DEMAND,
    /// @brief select data supermode in NETEDIT
    MID_HOTKEY_F4_SUPERMODE_DATA,
    /// @brief compute Network in network mode and Demand elements in demand mode
    MID_HOTKEY_F5_COMPUTE_NETWORK_DEMAND,
    /// @brief clean junctions without edges in network mode and unused routes in demand mode
    MID_HOTKEY_F6_CLEAN_SOLITARYJUNCTIONS_UNUSEDROUTES,
    /// @brief join selected junctions in network mode and normalice demand element ids in demand mode
    MID_HOTKEY_F7_JOIN_SELECTEDJUNCTIONS_ROUTES,
    /// @brief clean invalid crossings in network mode and demand elements in demand mode
    MID_HOTKEY_F8_CLEANINVALID_CROSSINGS_DEMANDELEMENTS,
    /// @brief open edit scheme menu
    MID_HOTKEY_F9_EDIT_VIEWSCHEME,
    /// @brief open options menu
    MID_HOTKEY_F10_OPTIONSMENU,
    /// @brief set/clear front element
    MID_HOTKEY_F11_FRONTELEMENT,
    /// @brief show the hall of fame dialog
    MID_HOTKEY_SHIFT_F11_HALLOFFAME,
    /// @brief open about dialog
    MID_HOTKEY_F12_ABOUT,
    /// @}

    /// @name shift + Functional hotkeys
    /// @{

    /// @brief set template
    MID_HOTKEY_SHIFT_F1_TEMPLATE_SET,
    /// @brief copy template
    MID_HOTKEY_SHIFT_F2_TEMPLATE_COPY,
    /// @brief clear template
    MID_HOTKEY_SHIFT_F3_TEMPLATE_CLEAR,
    /// @brief compute junctions with volatile options
    MID_HOTKEY_SHIFT_F5_COMPUTEJUNCTIONS_VOLATILE,
    /// @brief Adjust person plans (start und end positions, arrival positions, etc.)
    MID_HOTKEY_SHIFT_F7_ADJUST_PERSON_PLANS,
    /// @brief focus upper element of current frame (only used in NETEDIT)
    MID_HOTKEY_SHIFT_F12_FOCUSUPPERELEMENT,
    /// @}

    /// @name Other hotkeys
    /// @{

    /// @brief hot key <ESC> abort current edit operation
    MID_HOTKEY_ESC,
    /// @brief hot key <DEL> delete selections or elements
    MID_HOTKEY_DEL,
    /// @brief hot key <ENTER> accept current operation
    MID_HOTKEY_ENTER,
    /// @brief hot key <Backspace> remove last sub-operation
    MID_HOTKEY_BACKSPACE,

    /// @}

    /// @name application specific
    /// @{

    /// @brief Main window-ID
    MID_WINDOW,

    /// @}

    /// @name for common buttons
    /// @{

    /// @brief Ok-button pressed
    MID_OK,
    /// @brief Cancel-button pressed
    MID_CANCEL,
    /// @brief Update-button pressed
    MID_UPDATE,
    /// @}

    /// @name Main Window File Menu - IDs
    /// @{

    /// @brief Open configuration - ID
    MID_OPEN_CONFIG,
    /// @brief Open network - ID
    MID_OPEN_NETWORK,
    /// @brief Load additional file with poi and polygons
    MID_HOTKEY_CTRL_P,
    /// @brief Load edge data for visualization
    MID_OPEN_EDGEDATA,
    /// @brief Loads a file previously loaded
    MID_RECENTFILE,

    /// @}

    /// @name Main Window Simulation Control - IDs
    /// @{

    /// @brief Save state to file
    MID_SIMSAVE,
    /// @brief (quick)-load state from file
    MID_SIMLOAD,

    /// @}

    /// @name Main window windows control - IDs
    /// @{

    /// @brief Open a new microscopic view
    MID_NEW_MICROVIEW,
    /// @brief Open a new microscopic 3D view
    MID_NEW_OSGVIEW,

    /// @}

    /// @name SubThread - IDs
    /// @{

    /// @brief The loading thread
    ID_LOADTHREAD_EVENT,
    /// @brief The Simulation execution thread
    ID_RUNTHREAD_EVENT,
    /// @brief ID for message passing between threads
    ID_THREAD_EVENT,

    /// @}

    /// @name Edit Menu Item - IDs
    /// @{

    /// @brief Open editor for selections
    MID_EDITCHOSEN,
    /// @brief Open in netedit
    MID_NETEDIT,

    /// @}

    /// @name Settings Menu Item - IDs
    /// @{

    /// @brief Application settings - menu entry
    MID_APPSETTINGS,
    /// @brief Locator configuration - menu entry
    MID_LISTINTERNAL,
    MID_LISTPARKING,
    MID_LISTTELEPORTING,
    /// @brief The Simulation delay control
    MID_SIMDELAY,

    /// @}

    /// @brief Clear simulation output
    MID_CLEARMESSAGEWINDOW,
    /// @brief Show network statistics
    MID_SHOWNETSTATS,
    /// @brief Show vehicle statistics
    MID_SHOWVEHSTATS,
    /// @brief Show person statistics
    MID_SHOWPERSONSTATS,
    /// @brief update traci status
    MID_TRACI_STATUS,

    /// @name Common View Settings - IDs
    /// @{

    /// @brief Recenter view - button
    MID_RECENTERVIEW,
    /// @brief Allow rotation - button
    MID_ALLOWROTATION,
    /// @brief Locate junction - button
    MID_LOCATEJUNCTION,
    /// @brief Locate edge - button
    MID_LOCATEEDGE,
    /// @brief Locate vehicle - button
    MID_LOCATEVEHICLE,
    /// @brief Locate route - button
    MID_LOCATEROUTE,
    /// @brief Locate stop - button
    MID_LOCATESTOP,
    /// @brief Locate person - button
    MID_LOCATEPERSON,
    /// @brief Locate container - button
    MID_LOCATECONTAINER,
    /// @brief Locate TLS - button
    MID_LOCATETLS,
    /// @brief Locate additional structure - button
    MID_LOCATEADD,
    /// @brief Locate poi - button
    MID_LOCATEPOI,
    /// @brief Locate polygons - button
    MID_LOCATEPOLY,
    /// @brief Show tool tips - button
    MID_SHOWTOOLTIPS,
    /// @brief Make snapshot - button
    MID_MAKESNAPSHOT,
    /// @brief toggle zooming style
    MID_ZOOM_STYLE,
    /// @brief toggle time display mode
    MID_TIME_TOGGLE,
    /// @brief increase sim delay
    MID_DELAY_INC,
    /// @brief decrease sim delay
    MID_DELAY_DEC,
    /// @brief toggle delay between alternative value
    MID_DELAY_TOGGLE,
    /// @brief scale traffic
    MID_DEMAND_SCALE,
    /// @brief scale vehicle speed
    MID_SPEEDFACTOR,
    /// @}

    /// @name Common view IDs
    /// @{

    /// @brief GLCanvas - ID
    MID_GLCANVAS,
    /// @brief chore
    MID_CHORE,

    /// @}


    /// @name View settings - IDs
    /// @{

    /// @brief Change coloring scheme - combo
    MID_COLOURSCHEMECHANGE,

    /// @}

    /// @name Object PopUp Entries - IDs
    /// @{

    /// @brief Center view to object - popup entry
    MID_CENTER,
    /// @brief Copy object name - popup entry
    MID_COPY_NAME,
    /// @brief Copy typed object name - popup entry
    MID_COPY_TYPED_NAME,
    /// @brief Copy edge name (for lanes only)
    MID_COPY_EDGE_NAME,
    /// @brief Copy cursor position - popup entry
    MID_COPY_CURSOR_POSITION,
    /// @brief Copy cursor geo-coordinate position - popup entry
    MID_COPY_CURSOR_GEOPOSITION,
    /// @brief Show the cursor geo-coordinate position online in GeoHack - popup entry
    MID_SHOW_GEOPOSITION_ONLINE,
    /// @brief open additional dialog (used in netedit)
    MID_OPEN_ADDITIONAL_DIALOG,
    /// @brief Show object parameter - popup entry
    MID_SHOWPARS,
    /// @brief Show object type parameter - popup entry
    MID_SHOWTYPEPARS,
    /// @brief Show transportable plan
    MID_SHOWPLAN,
    /// @brief Show TLS phases - popup entry
    MID_SWITCH_OFF,
    MID_SWITCH,
    MID_SHOWPHASES = MID_SWITCH + 20,
    /// @brief Begin to track phases - menu entry
    MID_TRACKPHASES,
    /// @brief show/hide actuation detectors
    MID_SHOW_DETECTORS,
    /// @brief Add to selected items - menu entry
    MID_ADDSELECT,
    /// @brief Remove from selected items - Menu Entry
    MID_REMOVESELECT,
    /// @brief Open the object's manipulator
    MID_MANIP,
    /// @brief Draw a vehicle's route
    MID_DRAWROUTE,
    /// @brief Show vehicle's current route
    MID_SHOW_CURRENTROUTE,
    /// @brief Hide vehicle's current route
    MID_HIDE_CURRENTROUTE,
    /// @brief Show vehicle's future route
    MID_SHOW_FUTUREROUTE,
    /// @brief Hide vehicle's future route
    MID_HIDE_FUTUREROUTE,
    /// @brief Show vehicle's future route (without loops)
    MID_SHOW_ROUTE_NOLOOPS,
    /// @brief Hide vehicle's future route (without loops)
    MID_HIDE_ROUTE_NOLOOPS,
    /// @brief Show vehicle's best lanes
    MID_SHOW_BEST_LANES,
    /// @brief Hide vehicle's best lanes
    MID_HIDE_BEST_LANES,
    /// @brief Show all vehicle's routes
    MID_SHOW_ALLROUTES,
    /// @brief Hide all vehicle's routes
    MID_HIDE_ALLROUTES,
    /// @brief Start to track a vehicle
    MID_START_TRACK,
    /// @brief Stop to track a vehicle
    MID_STOP_TRACK,
    /// @brief select foes of a vehicle
    MID_SHOW_FOES,
    MID_SHOW_LFLINKITEMS,
    MID_HIDE_LFLINKITEMS,
    /// @brief select transportables of a vehicle
    MID_SELECT_TRANSPORTED,
    /// @brief toggle stop state of a vehicle or person
    MID_TOGGLE_STOP,
    /// @brief remove a vehicle or person
    MID_REMOVE_OBJECT,
    /// @brief Show persons's path on walkingarea
    MID_SHOW_WALKINGAREA_PATH,
    /// @brief Hide persons's path on walkingarea
    MID_HIDE_WALKINGAREA_PATH,
    /// @brief show reachability from a given lane
    MID_REACHABILITY,

    /// @}

    /// @name IDs used by Trackers
    /// @{

    /// @brief The Table
    MID_TABLE,
    /// @brief A Simulation step was performed
    MID_SIMSTEP,
    /// @brief A Tracker shall be opened
    MID_OPENTRACKER,

    /// @}

    /// @name General Setting Dialogs - IDs
    /// @{

    /// @brief Ok-button was pushed
    MID_SETTINGS_OK,
    /// @brief Cancel-button was pushed
    MID_SETTINGS_CANCEL,

    /// @}

    /// @name Application Settings - Dialog IDs
    /// @{

    /// @brief Close simulation at end - Option
    MID_QUITONSIMEND,
    /// @brief Start simulation when loaded - Option
    MID_AUTOSTART,
    /// @brief Demo mode - Option
    MID_DEMO,
    /// @brief Allow textures - Option
    MID_ALLOWTEXTURES,
    /// @brief Locate links in messages - Option
    MID_LOCATELINKS,
    /// @brief Set breakpionts from messages - Option
    MID_TIMELINK_BREAKPOINT,

    /// @}

    /// @name GLObjChooser - Dialog IDs
    /// @{

    /// @brief Center object
    MID_CHOOSER_CENTER,
    /// @brief Track object
    MID_CHOOSER_TRACK,
    /// @brief Text entry
    MID_CHOOSER_TEXT,
    /// @brief Object list
    MID_CHOOSER_LIST,
    /// @brief Filter selected
    MID_CHOOSER_FILTER,
    /// @brief Filter list by substring
    MID_CHOOSER_FILTER_SUBSTR,

    /// @}

    /// @name GLChosenEditor - Dialog IDs
    /// @{

    /// @brief set type of selection
    MID_CHOOSEN_OPERATION,
    /// @brief set subset of elements
    MID_CHOOSEN_ELEMENTS,
    /// @brief Load set
    MID_CHOOSEN_LOAD,
    /// @brief Save set
    MID_CHOOSEN_SAVE,
    /// @brief Clear set
    MID_CHOOSEN_CLEAR,
    /// @brief delete set
    MID_CHOOSEN_DELETE,
    /// @brief Reset set
    MID_CHOOSEN_RESET,
    /// @brief select all items
    MID_CHOOSEN_SELECT,
    /// @brief Deselect selected items
    MID_CHOOSEN_DESELECT,
    /// @brief Deselect selected items
    MID_CHOOSEN_INVERT,
    /// @brief Deselect selected items
    MID_CHOOSEN_NAME,
    /// @brief simplify network reduction
    MID_CHOOSEN_REDUCE,

    /// @}

    /// @name IDs for visualization settings - Dialog
    /// @{

    /// @brief Informs the dialog about a value's change
    MID_SIMPLE_VIEW_COLORCHANGE,
    /// @brief Informs the dialog about switching to another scheme
    MID_SIMPLE_VIEW_NAMECHANGE,
    /// @brief For the save-to-db - button
    MID_SIMPLE_VIEW_SAVE,
    /// @brief For the delete - button
    MID_SIMPLE_VIEW_DELETE,
    /// @brief For the export-to-file - button
    MID_SIMPLE_VIEW_EXPORT,
    /// @brief For the import-from-file - button
    MID_SIMPLE_VIEW_IMPORT,
    /// @brief For the load-decals - button
    MID_SIMPLE_VIEW_LOAD_DECALS,
    /// @brief For the save-decals - button
    MID_SIMPLE_VIEW_SAVE_DECALS,
    /// @brief For the clear-decals - button
    MID_SIMPLE_VIEW_CLEAR_DECALS,

    /// @}

    ///@brief help button
    MID_HELP,

    /// @name dynamic interaction with the simulation via SUMO-GUI
    /// @{

    /// @brief close lane
    MID_CLOSE_LANE,
    /// @brief close edge
    MID_CLOSE_EDGE,
    /// @brief add rerouter
    MID_ADD_REROUTER,
    /// @brief toggle detector override
    MID_VIRTUAL_DETECTOR,

    /// @}

    /// @name groupBoxModule messages
    /// @{
    /// @brief collapse groupBoxModule
    MID_GROUPBOXMODULE_COLLAPSE,
    /// @brief extends groupBoxModule
    MID_GROUPBOXMODULE_EXTEND,
    /// @brief reset widh groupBoxModule
    MID_GROUPBOXMODULE_RESETWIDTH,
    /// @brief save contents
    MID_GROUPBOXMODULE_SAVE,
    /// @brief load contents
    MID_GROUPBOXMODULE_LOAD,
    /// @}

    /// @name Toolbar file messages
    /// @{

    /// @brief open foreign network
    MID_GNE_TOOLBARFILE_OPENFOREIGN,
    /// @brief save all element
    MID_GNE_SAVEALLELEMENTS,
    /// @brief save joined junctions
    MID_GNE_SAVEJOINEDJUNCTIONS,
    /// @brief save additionals as
    MID_GNE_TOOLBARFILE_SAVEADDITIONALS_AS,
    /// @brief reload TLS Programs
    MID_GNE_TOOLBARFILE_RELOAD_TLSPROGRAMS,
    /// @brief reload edge types
    MID_GNE_TOOLBARFILE_RELOAD_EDGETYPES,
    /// @brief reload additionals
    MID_GNE_TOOLBARFILE_RELOAD_ADDITIONALS,
    /// @brief save TLS Programs as
    MID_GNE_TOOLBARFILE_SAVETLSPROGRAMS_AS,
    /// @brief save edgeTypes as
    MID_GNE_TOOLBARFILE_SAVEEDGETYPES_AS,
    /// @brief save demand elements as
    MID_GNE_TOOLBARFILE_SAVEDEMAND_AS,
    /// @brief reload demand elements
    MID_GNE_TOOLBARFILE_RELOAD_DEMANDELEMENTS,
    /// @brief save data elements as
    MID_GNE_TOOLBARFILE_SAVEDATA_AS,
    /// @brief reload data elements
    MID_GNE_TOOLBARFILE_RELOAD_DATAELEMENTS,

    /// @}

    /// @name Toolbar file messages
    /// @{

    /// @brief call tool netdiff
    MID_GNE_TOOLBARTOOLS_NETDIFF,

    /// @}

    /// @name Toolbar windows messages
    /// @{

    /// @brief load additionals in SUMO-GUI after press ctrl+T
    MID_GNE_TOOLBAREDIT_LOADADDITIONALS,
    /// @brief load demand in SUMO-GUI after press ctrl+T
    MID_GNE_TOOLBAREDIT_LOADDEMAND,
    /// @brief compute path manager
    MID_GNE_TOOLBAREDIT_COMPUTEPATHMANAGER,

    /// @}

    /// @name GNEViewParent dialog messages
    /// @{

    /// @brief Size of frame area updated
    MID_GNE_VIEWPARENT_FRAMEAREAWIDTH,

    /// @}

    /// @name GNEViewNet Network view options messages
    /// @{

    /// @brief show grid
    MID_GNE_NETWORKVIEWOPTIONS_TOGGLEGRID,
    /// @brief toggle draw junction shape
    MID_GNE_NETWORKVIEWOPTIONS_TOGGLEDRAWJUNCTIONSHAPE,
    /// @brief Draw vehicles in begin position or spread in lane
    MID_GNE_NETWORKVIEWOPTIONS_DRAWSPREADVEHICLES,
    /// @brief show demand elements
    MID_GNE_NETWORKVIEWOPTIONS_SHOWDEMANDELEMENTS,
    /// @brief select edges
    MID_GNE_NETWORKVIEWOPTIONS_SELECTEDGES,
    /// @brief show connections
    MID_GNE_NETWORKVIEWOPTIONS_SHOWCONNECTIONS,
    /// @brief hide connections
    MID_GNE_NETWORKVIEWOPTIONS_HIDECONNECTIONS,
    /// @brief show sub-additionals
    MID_GNE_NETWORKVIEWOPTIONS_SHOWSUBADDITIONALS,
    /// @brief show TAZ elements
    MID_GNE_NETWORKVIEWOPTIONS_SHOWTAZELEMENTS,
    /// @brief extend selection
    MID_GNE_NETWORKVIEWOPTIONS_EXTENDSELECTION,
    /// @brief change all phases
    MID_GNE_NETWORKVIEWOPTIONS_CHANGEALLPHASES,
    /// @brief ask before merging junctions
    MID_GNE_NETWORKVIEWOPTIONS_ASKFORMERGE,
    /// @brief show junctions as bubbles
    MID_GNE_NETWORKVIEWOPTIONS_SHOWBUBBLES,
    /// @brief move elevation instead of x,y
    MID_GNE_NETWORKVIEWOPTIONS_MOVEELEVATION,
    /// @brief create edges in chain mode
    MID_GNE_NETWORKVIEWOPTIONS_CHAINEDGES,
    /// @brief automatically create opposite edge
    MID_GNE_NETWORKVIEWOPTIONS_AUTOOPPOSITEEDGES,
    /// @brief Add edge to selected items - menu entry
    MID_GNE_ADDSELECT_EDGE,
    /// @brief Remove edge from selected items - Menu Entry
    MID_GNE_REMOVESELECT_EDGE,
    /// @brief lock element
    MID_GNE_LOCK_ELEMENT,
    /// @brief lock all element
    MID_GNE_LOCK_ALLELEMENTS,
    /// @brief unlock all element
    MID_GNE_UNLOCK_ALLELEMENTS,
    /// @brief lock selected element
    MID_GNE_LOCK_SELECTEDELEMENTS,

    /// @}

    /// @name GNEViewNet Demand view options messages
    /// @{

    /// @brief show grid
    MID_GNE_DEMANDVIEWOPTIONS_SHOWGRID,
    /// @brief toggle draw junction shape
    MID_GNE_DEMANDVIEWOPTIONS_TOGGLEDRAWJUNCTIONSHAPE,
    /// @brief Draw vehicles in begin position or spread in lane
    MID_GNE_DEMANDVIEWOPTIONS_DRAWSPREADVEHICLES,
    /// @brief hide non-inspected demand element
    MID_GNE_DEMANDVIEWOPTIONS_HIDENONINSPECTED,
    /// @brief hide shapes
    MID_GNE_DEMANDVIEWOPTIONS_HIDESHAPES,
    /// @brief show all trips
    MID_GNE_DEMANDVIEWOPTIONS_SHOWTRIPS,
    /// @brief show all person plans
    MID_GNE_DEMANDVIEWOPTIONS_SHOWALLPERSONPLANS,
    /// @brief lock person
    MID_GNE_DEMANDVIEWOPTIONS_LOCKPERSON,
    /// @brief show all container plans
    MID_GNE_DEMANDVIEWOPTIONS_SHOWALLCONTAINERPLANS,
    /// @brief lock container
    MID_GNE_DEMANDVIEWOPTIONS_LOCKCONTAINER,
    /// @brief show overlapped routes
    MID_GNE_DEMANDVIEWOPTIONS_SHOWOVERLAPPEDROUTES,

    /// @}

    /// @name GNEViewNet Data view options messages
    /// @{

    /// @brief toggle draw junction shape
    MID_GNE_DATAVIEWOPTIONS_TOGGLEDRAWJUNCTIONSHAPE,
    /// @brief show additionals
    MID_GNE_DATAVIEWOPTIONS_SHOWADDITIONALS,
    /// @brief show shapes
    MID_GNE_DATAVIEWOPTIONS_SHOWSHAPES,
    /// @brief show demand elements
    MID_GNE_DATAVIEWOPTIONS_SHOWDEMANDELEMENTS,
    /// @brief toggle TAZRel drawing
    MID_GNE_DATAVIEWOPTIONS_TAZRELDRAWING,
    /// @brief toggle draw TAZ fill
    MID_GNE_DATAVIEWOPTIONS_TAZDRAWFILL,
    /// @brief toggle draw TAZRel only from
    MID_GNE_DATAVIEWOPTIONS_TAZRELONLYFROM,
    /// @brief toggle draw TAZRel only to
    MID_GNE_DATAVIEWOPTIONS_TAZRELONLYTO,

    /// @}

    /// @name GNEViewNet interval bar messages
    /// @{

    /// @brief generic data selected
    MID_GNE_INTERVALBAR_GENERICDATATYPE,
    /// @brief data set selected
    MID_GNE_INTERVALBAR_DATASET,
    /// @brief enable/disable show data elements by interval
    MID_GNE_INTERVALBAR_LIMITED,
    /// @brief begin changed in InterbalBar
    MID_GNE_INTERVALBAR_BEGIN,
    /// @brief end changed in InterbalBar
    MID_GNE_INTERVALBAR_END,
    /// @brief parameter changed in InterbalBar
    MID_GNE_INTERVALBAR_PARAMETER,

    /// @}

    /// @name GNEFrame messages
    /// @{

    // MID_GNE_WIZARD,
    /// @brief attribute added
    MID_GNE_ADD_ATTRIBUTE,
    /// @brief attribute removed
    MID_GNE_REMOVE_ATTRIBUTE,
    /// @brief attribute edited
    MID_GNE_SET_ATTRIBUTE,
    /// @brief create element
    MID_GNE_CREATE,
    /// @brief delete element
    MID_GNE_DELETE,
    /// @brief inspect element
    MID_GNE_INSPECT,
    /// @brief reset element
    MID_GNE_RESET,
    /// @brief apply element
    MID_GNE_APPLY,
    /// @brief center element
    MID_GNE_CENTER,
    /// @brief copy element
    MID_GNE_COPY,
    /// @brief select element
    MID_GNE_SELECT,
    /// @brief used to select a type of element in a combo box
    MID_GNE_SET_TYPE,
    /// @brief tag type selected in ComboBox
    MID_GNE_TAGTYPE_SELECTED,
    /// @brief tag selected in ComboBox
    MID_GNE_TAG_SELECTED,
    /// @brief create new data set
    MID_GNE_DATASET_NEW,
    /// @brief GNEDataInterval selected in comboBox of IntervalBar
    MID_GNE_DATASET_SELECTED,
    /// @brief GNEDataInterval selected in comboBox of IntervalBar
    MID_GNE_DATAINTERVAL_SELECTED,
    /// @brief text attribute edited
    MID_GNE_SET_ATTRIBUTE_TEXT,
    /// @brief bool attribute edited
    MID_GNE_SET_ATTRIBUTE_BOOL,
    /// @brief attribute edited trough dialog
    MID_GNE_SET_ATTRIBUTE_DIALOG,
    /// @brief open parameters dialog
    MID_GNE_OPEN_PARAMETERS_DIALOG,
    /// @brief attribute selected using button (radio button or checkbox)
    MID_GNE_SET_ATTRIBUTE_BUTTON,
    /// @brief abort edge path creation
    MID_GNE_PATHCREATOR_ABORT,
    /// @brief finish edge path creation
    MID_GNE_PATHCREATOR_FINISH,
    /// @brief use last inserted route
    MID_GNE_PATHCREATOR_USELASTROUTE,
    /// @brief remove last inserted element in path
    MID_GNE_PATHCREATOR_REMOVELAST,
    /// @brief enable or disable show path candidates
    MID_GNE_PATHCREATOR_SHOWCANDIDATES,
    /// @brief abort lane path creation
    MID_GNE_ABORT,
    /// @brief finish lane path creation
    MID_GNE_FINISH,
    /// @brief remove last inserted element in path
    MID_GNE_REMOVELAST,
    /// @brief enable or disable show path candidates
    MID_GNE_SHOWCANDIDATES,
    /// @brief In GNEElementTree list, show child menu
    MID_GNE_ACHIERARCHY_SHOWCHILDMENU,
    /// @brief In GNEElementTree list, move element to up
    MID_GNE_ACHIERARCHY_MOVEUP,
    /// @brief In GNEElementTree list, move element to down
    MID_GNE_ACHIERARCHY_MOVEDOWN,
    /// @brief start drawing polygon
    MID_GNE_STARTDRAWING,
    /// @brief stop drawing polygon
    MID_GNE_STOPDRAWING,
    /// @brief abort drawing polygon
    MID_GNE_ABORTDRAWING,
    /// @brief inspect next element in overlapped module
    MID_GNE_OVERLAPPED_NEXT,
    /// @brief inspect previous element in overlapped module
    MID_GNE_OVERLAPPED_PREVIOUS,
    /// @brief show list of overlapped elements
    MID_GNE_OVERLAPPED_SHOWLIST,
    /// @brief list item selected in overlapped module
    MID_GNE_OVERLAPPED_ITEMSELECTED,
    /// @brief mark item as front element
    MID_GNE_MARKFRONTELEMENT,
    /// @brief set custom geometry point
    MID_GNE_CUSTOM_GEOMETRYPOINT,
    /// @brief reset geometry point
    MID_GNE_RESET_GEOMETRYPOINT,
    /// @brief set zero fringe probabilities (used in TAZ Frame)
    MID_GNE_SET_ZEROFRINGEPROB,

    /// @}

    /// @name GNESelectorFrame messages
    /// @{

    /// @brief select tag in selector frame
    MID_GNE_SELECTORFRAME_SELECTTAG,
    /// @brief select attribute in selector frame
    MID_GNE_SELECTORFRAME_SELECTATTRIBUTE,
    /// @brief process string
    MID_GNE_SELECTORFRAME_PROCESSSTRING,
    /// @brief changes the visual scaling of selected items
    MID_GNE_SELECTORFRAME_SELECTSCALE,
    /// @brief change interval
    MID_GNE_SELECTORFRAME_SETINTERVAL,
    /// @brief edit begin text field
    MID_GNE_SELECTORFRAME_SETBEGIN,
    /// @brief end text field
    MID_GNE_SELECTORFRAME_SETEND,
    /// @brief select/unselect parents
    MID_GNE_SELECTORFRAME_PARENTS,
    /// @brief select/unselect children
    MID_GNE_SELECTORFRAME_CHILDREN,

    /// @}

    /// @name GNEConnectorFrame messages
    /// @{

    /// @brief select dead end lanes
    MID_GNE_CONNECTORFRAME_SELECTDEADENDS,
    /// @brief select lanes that have no connection leading to it
    MID_GNE_CONNECTORFRAME_SELECTDEADSTARTS,
    /// @brief select lanes that are connected from concurrent lanes
    MID_GNE_CONNECTORFRAME_SELECTCONFLICTS,
    /// @brief select lanes with connections that have the pass attribute set to 'true'
    MID_GNE_CONNECTORFRAME_SELECTPASS,

    /// @}

    /// @name GNEInspectorFrame messages
    /// @{

    /// @brief go back to the previous element
    MID_GNE_INSPECTORFRAME_GOBACK,

    /// @}

    /// @name GNECreateEdgeFrame messages
    /// @{
    /// @brief selected radio button
    MID_GNE_CREATEEDGEFRAME_SELECTRADIOBUTTON,
    /// @brief add new edge type
    MID_GNE_CREATEEDGEFRAME_ADD,
    /// @brief delete edge type
    MID_GNE_CREATEEDGEFRAME_DELETE,
    /// @brief delete edge type
    MID_GNE_CREATEEDGEFRAME_SELECT,
    /// @brief create edge type from template
    MID_GNE_CREATEEDGEFRAME_CREATEFROMTEMPLATE,

    /// @}

    /// @name GNETLSEditorFrame messages
    /// @{

    /// @brief selected junction von TLS
    MID_GNE_TLSFRAME_SELECT_JUNCTION,
    /// @brief update TLS status
    MID_GNE_TLSFRAME_UPDATE_STATUS,
    /// @brief replace program with a newly guessed program
    MID_GNE_TLSFRAME_GUESSPROGRAM,
    /// @brief TLS offset
    MID_GNE_TLSFRAME_OFFSET,
    /// @brief TLS parameters
    MID_GNE_TLSFRAME_PARAMETERS,
    /// @brief switch between programs
    MID_GNE_TLSFRAME_SWITCH,
    /// @brief Create TLS
    MID_GNE_TLSFRAME_CREATE,
    /// @brief delete TLS
    MID_GNE_TLSFRAME_DELETE,
    /// @brief regenerate TLS
    MID_GNE_TLSFRAME_REGENERATE,
    /// @brief rename TLS
    MID_GNE_TLSFRAME_RENAME,
    /// @brief sub-rename TLS
    MID_GNE_TLSFRAME_SUBRENAME,
    /// @brief add off to TLS
    MID_GNE_TLSFRAME_ADDOFF,
    /// @brief select phase thable
    MID_GNE_TLSFRAME_PHASE_TABLE,
    /// @brief create phase thable
    MID_GNE_TLSFRAME_PHASE_CREATE,
    /// @brief delete  phase thable
    MID_GNE_TLSFRAME_PHASE_DELETE,
    /// @brief cleanup unused states
    MID_GNE_TLSFRAME_CLEANUP,
    /// @brief mark unused states
    MID_GNE_TLSFRAME_MARKUNUSED,
    /// @brief add unused states
    MID_GNE_TLSFRAME_ADDUNUSED,
    /// @brief group states
    MID_GNE_TLSFRAME_GROUP_STATES,
    /// @brief ungroup states
    MID_GNE_TLSFRAME_UNGROUP_STATES,
    /// @brief Load Program
    MID_GNE_TLSFRAME_LOAD_PROGRAM,
    /// @brief cleanup unused states
    MID_GNE_TLSFRAME_SAVE_PROGRAM,

    /// @}

    /// @name GNEAdditionalFrame messages
    /// @{

    /// @brief search element
    MID_GNE_SEARCH,
    /// @brief use selected elements
    MID_GNE_USESELECTED,
    /// @brief clear selection of elements
    MID_GNE_CLEARSELECTION,
    /// @brief invert selection of elements
    MID_GNE_INVERTSELECTION,
    /// @brief stop selection of consecutive edges/lanes
    MID_GNE_STOPSELECTION,
    /// @brief abort selection of consecutive edges/lanes
    MID_GNE_ABORTSELECTION,

    /// @}

    /// @name GNEProhibitionFrame messages
    /// @{

    /// @brief select prohibiting connection
    MID_GNE_PROHIBITIONFRAME_SELECTPROHIBITING,
    /// @brief select connections to be prohibited
    MID_GNE_PROHIBITIONFRAME_SELECTPROHIBITED,
    /// @brief abort drawing polygon
    MID_GNE_PROHIBITIONFRAME_ABORTDRAWING,

    /// @}

    /// @name GNERoute messages
    /// @{

    /// @brief select a route mode
    MID_GNE_ROUTEFRAME_ROUTEMODE,
    /// @brief select a VClass
    MID_GNE_ROUTEFRAME_VCLASS,

    /// @}

    /// @name GNEEdge messages
    /// @{

    /// @brief change default geometry endpoints
    MID_GNE_EDGE_EDIT_ENDPOINT,
    /// @brief reset default geometry endpoints
    MID_GNE_EDGE_RESET_ENDPOINT,
    /// @brief remove inner geometry
    MID_GNE_EDGE_STRAIGHTEN,
    /// @brief smooth geometry
    MID_GNE_EDGE_SMOOTH,
    /// @brief interpolate z values linear between junctions
    MID_GNE_EDGE_STRAIGHTEN_ELEVATION,
    /// @brief smooth elevation with regard to adjoining edges
    MID_GNE_EDGE_SMOOTH_ELEVATION,
    /// @brief split an edge
    MID_GNE_EDGE_SPLIT,
    /// @brief split an edge
    MID_GNE_EDGE_SPLIT_BIDI,
    /// @brief reverse an edge
    MID_GNE_EDGE_REVERSE,
    /// @brief add reverse edge
    MID_GNE_EDGE_ADD_REVERSE,
    /// @brief reset custom lengths
    MID_GNE_EDGE_RESET_LENGTH,

    /// @}

    /// @name GNEJunction messages
    /// @{

    /// @brief clear junction's connections
    MID_GNE_JUNCTION_CLEAR_CONNECTIONS,
    /// @brief reset junction's connections
    MID_GNE_JUNCTION_RESET_CONNECTIONS,
    /// @brief turn junction into geometry node
    MID_GNE_JUNCTION_REPLACE,
    /// @brief turn junction into multiple junctions
    MID_GNE_JUNCTION_SPLIT,
    /// @brief turn junction into multiple junctions and reconnect them heuristically
    MID_GNE_JUNCTION_SPLIT_RECONNECT,
    /// @brief edit junction shape
    MID_GNE_JUNCTION_EDIT_SHAPE,
    /// @brief reset junction shape
    MID_GNE_JUNCTION_RESET_SHAPE,
    /// @brief select all roundabout nodes and edges of the current roundabout
    MID_GNE_JUNCTION_SELECT_ROUNDABOUT,
    /// @brief convert junction to roundabout
    MID_GNE_JUNCTION_CONVERT_ROUNDABOUT,

    /// @}

    /// @name GNEConnection messages
    /// @{

    /// @brief edit connection shape
    MID_GNE_CONNECTION_EDIT_SHAPE,
    /// @ brief smooth connection shape
    MID_GNE_CONNECTION_SMOOTH_SHAPE,

    /// @}


    /// @name GNECrossing messages
    /// @{

    /// @brief edit crossing shape
    MID_GNE_CROSSING_EDIT_SHAPE,

    /// @}


    /// @name GNEPoly messages
    /// @{

    /// @brief simplify polygon geometry
    MID_GNE_POLYGON_SIMPLIFY_SHAPE,
    /// @brief close opened polygon
    MID_GNE_POLYGON_CLOSE,
    /// @brief open closed polygon
    MID_GNE_POLYGON_OPEN,
    /// @brief Set a vertex of polygon as first vertex
    MID_GNE_POLYGON_SET_FIRST_POINT,
    /// @brief delete geometry point
    MID_GNE_POLYGON_DELETE_GEOMETRY_POINT,
    /// @brief select elements within polygon boundary
    MID_GNE_POLYGON_SELECT,

    /// @}

    /// @name GNEPOI messages
    /// @{

    /// @brief Transform POI to POILane, and viceversa
    MID_GNE_POI_TRANSFORM,

    /// @}

    /// @name GNELane messages
    /// @{

    /// @brief duplicate a lane
    MID_GNE_LANE_DUPLICATE,
    /// @brief edit lane shape
    MID_GNE_LANE_EDIT_SHAPE,
    /// @brief reset custom shape
    MID_GNE_LANE_RESET_CUSTOMSHAPE,
    /// @brief reset opposite lane
    MID_GNE_LANE_RESET_OPPOSITELANE,
    /// @brief transform lane to sidewalk
    MID_GNE_LANE_TRANSFORM_SIDEWALK,
    /// @brief transform lane to bikelane
    MID_GNE_LANE_TRANSFORM_BIKE,
    /// @brief transform lane to busLane
    MID_GNE_LANE_TRANSFORM_BUS,
    /// @brief transform lane to greenVerge
    MID_GNE_LANE_TRANSFORM_GREENVERGE,
    /// @brief add sidewalk
    MID_GNE_LANE_ADD_SIDEWALK,
    /// @brief add bikelane
    MID_GNE_LANE_ADD_BIKE,
    /// @brief add busLane
    MID_GNE_LANE_ADD_BUS,
    /// @brief add greenVerge front of current lane
    MID_GNE_LANE_ADD_GREENVERGE_FRONT,
    /// @brief add greenVerge back of current lane
    MID_GNE_LANE_ADD_GREENVERGE_BACK,
    /// @brief remove sidewalk
    MID_GNE_LANE_REMOVE_SIDEWALK,
    /// @brief remove bikelane
    MID_GNE_LANE_REMOVE_BIKE,
    /// @brief remove busLane
    MID_GNE_LANE_REMOVE_BUS,
    /// @brief remove greenVerge
    MID_GNE_LANE_REMOVE_GREENVERGE,

    /// @}

    /// @name GNEDemandElements messages
    /// @{

    /// @brief apply distance
    MID_GNE_ROUTE_APPLY_DISTANCE,
    /// @brief transform vehicle to another vehicle type (ej: flow to trip)
    MID_GNE_VEHICLE_TRANSFORM,
    /// @brief transform person to another person type (ej: person to personflow)
    MID_GNE_PERSON_TRANSFORM,
    /// @brief transform container to another container type (ej: container to containerflow)
    MID_GNE_CONTAINER_TRANSFORM,
    /// @}

    /// @name Netedit Dialogs
    /// @{

    /// @brief accept button
    MID_GNE_BUTTON_ACCEPT,
    /// @brief cancel button
    MID_GNE_BUTTON_CANCEL,
    /// @brief reset button
    MID_GNE_BUTTON_RESET,
    /// @brief load button
    MID_GNE_BUTTON_LOAD,
    /// @brief save button
    MID_GNE_BUTTON_SAVE,
    /// @brief clear button
    MID_GNE_BUTTON_CLEAR,
    /// @brief sort button
    MID_GNE_BUTTON_SORT,
    /// @brief focus button
    MID_GNE_BUTTON_FOCUS,

    /// @}

    /// @name Variable Speed Sign Dialog messages
    /// @{

    /// @brief add row
    MID_GNE_VARIABLESPEEDSIGN_ADDROW,
    /// @brief Click over Table
    MID_GNE_VARIABLESPEEDSIGN_TABLE,
    /// @brief sort table values
    MID_GNE_VARIABLESPEEDSIGN_SORT,

    /// @}

    /// @name Rerouter Dialog messages
    /// @{

    /// @brief select table interval
    MID_GNE_REROUTEDIALOG_TABLE_INTERVAL,
    /// @brief change table closing lane reroute reroute
    MID_GNE_REROUTEDIALOG_TABLE_CLOSINGLANEREROUTE,
    /// @brief change table route closing reroute
    MID_GNE_REROUTEDIALOG_TABLE_CLOSINGREROUTE,
    /// @brief change table destiny probability reroute
    MID_GNE_REROUTEDIALOG_TABLE_DESTPROBREROUTE,
    /// @brief change table route probability reroute
    MID_GNE_REROUTEDIALOG_TABLE_ROUTEPROBREROUTE,
    /// @brief change table parkingAreaReroute
    MID_GNE_REROUTEDIALOG_TABLE_PARKINGAREAREROUTE,
    /// @brief add interval
    MID_GNE_REROUTEDIALOG_ADD_INTERVAL,
    /// @brief sort rerouter intervals
    MID_GNE_REROUTEDIALOG_SORT_INTERVAL,
    /// @brief add closing lane reroute
    MID_GNE_REROUTEDIALOG_ADD_CLOSINGLANEREROUTE,
    /// @brief add closing reroute
    MID_GNE_REROUTEDIALOG_ADD_CLOSINGREROUTE,
    /// @brief add destiny probability route
    MID_GNE_REROUTEDIALOG_ADD_DESTPROBREROUTE,
    /// @brief add route probability route
    MID_GNE_REROUTEDIALOG_ADD_ROUTEPROBREROUTE,
    /// @brief add parkingAreaReroute
    MID_GNE_REROUTEDIALOG_ADD_PARKINGAREAREROUTE,
    /// @brief edit interval
    MID_GNE_REROUTEDIALOG_EDIT_INTERVAL,

    /// @}

    /// @name Calibrator Dialog messages
    /// @{

    /// @brief change table route
    MID_GNE_CALIBRATORDIALOG_TABLE_ROUTE,
    /// @brief add new route
    MID_GNE_CALIBRATORDIALOG_ADD_ROUTE,
    /// @brief change table flow
    MID_GNE_CALIBRATORDIALOG_TABLE_FLOW,
    MID_GNE_CALIBRATORDIALOG_ADD_FLOW,
    /// @brief change table route
    MID_GNE_CALIBRATORDIALOG_TABLE_VEHICLETYPE,
    /// @brief add vehicle type
    MID_GNE_CALIBRATORDIALOG_ADD_VEHICLETYPE,

    /// @}

    /// @name allowDisallow Dialog messages
    /// @{

    /// @brief select/unselect single vehicle
    MID_GNE_ALLOWDISALLOW_CHANGE,
    /// @brief select all vehicles
    MID_GNE_ALLOWDISALLOW_SELECTALL,
    /// @brief unselect all vehicles
    MID_GNE_ALLOWDISALLOW_UNSELECTALL,
    /// @brief select only non road vehicles
    MID_GNE_ALLOWDISALLOW_ONLY_ROAD,
    /// @brief select only rail vehicles
    MID_GNE_ALLOWDISALLOW_ONLY_RAIL,

    /// @}

    /// @name Fix Stoppingplaces Dialog messages
    /// @{

    /// @brief FixStoppingPlaces dialog
    MID_GNE_FIXSTOPPINGPLACES_CHANGE,

    /// @}

    /// @name other
    /// @{

    /// @brief selector for ModesMenuTitle
    MID_GNE_MODESMENUTITLE,
    /// @brief selector for LockMenuTitle
    MID_GNE_LOCK_MENUTITLE,
    /// @brief open undo list dialog
    MID_GNE_UNDOLISTDIALOG,
    /// @brief update undolist
    MID_GNE_UNDOLIST_UPDATE,

    /// @}

    /// @brief last element of enum (not used)
    MID_LAST
};
