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
/// @file    GNEViewNet.h
/// @author  Jakob Erdmann
/// @date    Feb 2011
///
// A view on the network being edited (adapted from GUIViewTraffic)
/****************************************************************************/
#pragma once
#include <config.h>

#include <utils/gui/windows/GUISUMOAbstractView.h>

#include "GNEViewNetHelper.h"


// ===========================================================================
// class declaration
// ===========================================================================
class GNEFrame;
class GNENet;
class GNEUndoList;
class GNEViewParent;

// ===========================================================================
// class definitions
// ===========================================================================
/**
 * @class GNEViewNet
 * Microsocopic view at the simulation
 */
class GNEViewNet : public GUISUMOAbstractView {
    /// @brief FOX-declaration
    FXDECLARE(GNEViewNet)

    /// @brief declare GNEViewNetHelper as friend struct
    friend struct GNEViewNetHelper;

public:
    /* @brief constructor
     * @param[in] tmpParent temporal FXFrame parent so that we can add items to view area in the desired order
     * @param[in] actualParent FXFrame parent of GNEViewNet
     * @param[in] app main windows
     * @param[in] viewParent viewParent of this viewNet
     * @param[in] net traffic net
     * @param[in] newNet check if we're creating a new net, or loading an existent
     * @param[in] undoList pointer to UndoList module
     * @param[in] glVis a reference to GLVisuals
     * @param[in] share a reference to FXCanvas
     */
    GNEViewNet(FXComposite* tmpParent, FXComposite* actualParent, GUIMainWindow& app,
               GNEViewParent* viewParent, GNENet* net, const bool newNet, GNEUndoList* undoList,
               FXGLVisual* glVis, FXGLCanvas* share);

    /// @brief destructor
    ~GNEViewNet();

    /// @brief recalculate boundaries
    void recalculateBoundaries();

    /// @brief builds the view toolbars
    void buildViewToolBars(GUIGlChildWindow* v);

    /// @brief Mark the entire GNEViewNet to be repainted later
    void updateViewNet() const;

    /// @brief set supermode Network (used after load/create new network)
    void forceSupermodeNetwork();

    /// @brief get AttributeCarriers in Boundary
    std::set<std::pair<std::string, GNEAttributeCarrier*> > getAttributeCarriersInBoundary(const Boundary& boundary, bool forceSelectEdges = false);

    /// @brief get objects under cursor
    const GNEViewNetHelper::ObjectsUnderCursor& getObjectsUnderCursor() const;

    /// @brief get move multiple element values
    const GNEViewNetHelper::MoveMultipleElementValues& getMoveMultipleElementValues() const;

    /** @brief Builds an entry which allows to (de)select the object
     * @param ret The popup menu to add the entry to
     * @param AC AttributeCarrier that will be select/unselected
     */
    void buildSelectionACPopupEntry(GUIGLObjectPopupMenu* ret, GNEAttributeCarrier* AC);

    /// @brief set color scheme
    bool setColorScheme(const std::string& name);

    ///@brief recalibrate color scheme according to the current value range
    void buildColorRainbow(const GUIVisualizationSettings& s, GUIColorScheme& scheme, int active, GUIGlObjectType objectType,
                           bool hide = false, double hideThreshold = 0.);

    /// @brief return list of available edge parameters
    std::vector<std::string> getEdgeLaneParamKeys(bool edgeKeys) const;

    /// @brief return list of loaded edgeData attributes
    std::vector<std::string> getEdgeDataAttrs() const;

    /// @brief return list of loaded edgeRelation and tazRelation attributes
    std::vector<std::string> getRelDataAttrs() const;

    /// @brief open object dialog
    void openObjectDialogAtCursor();

    // save visualization settings
    void saveVisualizationSettings() const;

    /// @brief get edit modes
    const GNEViewNetHelper::EditModes& getEditModes() const;

    /// @brief get testing mode
    const GNEViewNetHelper::TestingMode& getTestingMode() const;

    /// @brief get network view options
    const GNEViewNetHelper::NetworkViewOptions& getNetworkViewOptions() const;

    /// @brief get demand view options
    const GNEViewNetHelper::DemandViewOptions& getDemandViewOptions() const;

    /// @brief get data view options
    const GNEViewNetHelper::DataViewOptions& getDataViewOptions() const;

    /// @brief get Key Pressed module
    const GNEViewNetHelper::MouseButtonKeyPressed& getMouseButtonKeyPressed() const;

    /// @brief get Edit Shape module
    const GNEViewNetHelper::EditNetworkElementShapes& getEditNetworkElementShapes() const;

    /// @name overloaded handlers
    /// @{
    /// @brief called when user press mouse's left button
    long onLeftBtnPress(FXObject*, FXSelector, void*);

    /// @brief called when user releases mouse's left button
    long onLeftBtnRelease(FXObject*, FXSelector, void*);

    /// @brief called when user press mouse's right button
    long onRightBtnPress(FXObject*, FXSelector, void*);

    /// @brief called when user releases mouse's right button
    long onRightBtnRelease(FXObject*, FXSelector, void*);

    /// @brief called when user moves mouse
    long onMouseMove(FXObject*, FXSelector, void*);

    /// @brief called when user press a key
    long onKeyPress(FXObject* o, FXSelector sel, void* data);

    /// @brief called when user release a key
    long onKeyRelease(FXObject* o, FXSelector sel, void* data);
    /// @}

    /// @name set modes call backs
    /// @{
    /// @brief called when user press a supermode button
    long onCmdSetSupermode(FXObject*, FXSelector sel, void*);

    /// @brief called when user press a mode button (Network or demand)
    long onCmdSetMode(FXObject*, FXSelector sel, void*);

    /// @}

    /// @brief split edge at cursor position
    long onCmdSplitEdge(FXObject*, FXSelector, void*);

    /// @brief split edge at cursor position
    long onCmdSplitEdgeBidi(FXObject*, FXSelector, void*);

    /// @brief reverse edge
    long onCmdReverseEdge(FXObject*, FXSelector, void*);

    /// @brief add reversed edge
    long onCmdAddReversedEdge(FXObject*, FXSelector, void*);

    /// @brief change geometry endpoint
    long onCmdEditEdgeEndpoint(FXObject*, FXSelector, void*);

    /// @brief change geometry endpoint
    long onCmdResetEdgeEndpoint(FXObject*, FXSelector, void*);

    /// @brief makes selected edges straight
    long onCmdStraightenEdges(FXObject*, FXSelector, void*);

    /// @brief smooth geometry
    long onCmdSmoothEdges(FXObject*, FXSelector, void*);

    /// @brief interpolate z values linear between junctions
    long onCmdStraightenEdgesElevation(FXObject*, FXSelector, void*);

    /// @brief smooth elevation with regard to adjoining edges
    long onCmdSmoothEdgesElevation(FXObject*, FXSelector, void*);

    /// @brief reset custom edge lengths
    long onCmdResetLength(FXObject*, FXSelector, void*);

    /// @brief simply shape of current polygon
    long onCmdSimplifyShape(FXObject*, FXSelector, void*);

    /// @brief delete the closes geometry point
    long onCmdDeleteGeometryPoint(FXObject*, FXSelector, void*);

    /// @brief close opened polygon
    long onCmdClosePolygon(FXObject*, FXSelector, void*);

    /// @brief open closed polygon
    long onCmdOpenPolygon(FXObject*, FXSelector, void*);

    /// @brief select elements within polygon boundary
    long onCmdSelectPolygonElements(FXObject*, FXSelector, void*);

    /// @brief set as first geometry point the closes geometry point
    long onCmdSetFirstGeometryPoint(FXObject*, FXSelector, void*);

    /// @brief transform POI to POILane, and viceversa
    long onCmdTransformPOI(FXObject*, FXSelector, void*);

    /// @brief set custom geometry point
    long onCmdSetCustomGeometryPoint(FXObject*, FXSelector, void*);

    /// @brief reset edge end points
    long onCmdResetEndPoints(FXObject*, FXSelector, void*);

    /// @brief duplicate selected lane
    long onCmdDuplicateLane(FXObject*, FXSelector, void*);

    /// @brief edit lane shape
    long onCmdEditLaneShape(FXObject*, FXSelector, void*);

    /// @brief reset custom shapes of selected lanes
    long onCmdResetLaneCustomShape(FXObject*, FXSelector, void*);

    /// @brief reset oppositeLane of current lane
    long onCmdResetOppositeLane(FXObject*, FXSelector, void*);

    /// @brief add/remove/restrict lane
    long onCmdLaneOperation(FXObject*, FXSelector sel, void*);

    /// @brief show lane reachability
    long onCmdLaneReachability(FXObject*, FXSelector sel, void*);

    /// @brief open additional dialog
    long onCmdOpenAdditionalDialog(FXObject*, FXSelector, void*);

    /// @brief edit junction shape
    long onCmdEditJunctionShape(FXObject*, FXSelector, void*);

    /// @brief reset junction shape
    long onCmdResetJunctionShape(FXObject*, FXSelector, void*);

    /// @brief replace node by geometry
    long onCmdReplaceJunction(FXObject*, FXSelector, void*);

    /// @brief split junction into multiple junctions
    long onCmdSplitJunction(FXObject*, FXSelector, void*);

    /// @brief split junction into multiple junctions and reconnect them
    long onCmdSplitJunctionReconnect(FXObject*, FXSelector, void*);

    /// @brief select all roundabout nodes and edges
    long onCmdSelectRoundabout(FXObject*, FXSelector, void*);

    /// @brief convert junction to roundabout
    long onCmdConvertRoundabout(FXObject*, FXSelector, void*);

    /// @brief clear junction connections
    long onCmdClearConnections(FXObject*, FXSelector, void*);

    /// @brief reset junction connections
    long onCmdResetConnections(FXObject*, FXSelector, void*);

    /// @brief edit connection shape
    long onCmdEditConnectionShape(FXObject*, FXSelector, void*);

    /// @brief edit connection shape
    long onCmdSmoothConnectionShape(FXObject*, FXSelector, void*);

    /// @brief edit crossing shape
    long onCmdEditCrossingShape(FXObject*, FXSelector, void*);

    /// @name View options network call backs
    /// @{

    /// @brief toggle select edges
    long onCmdToggleSelectEdges(FXObject*, FXSelector, void*);

    /// @brief toggle show connections
    long onCmdToggleShowConnections(FXObject*, FXSelector, void*);

    /// @brief toggle hide connections
    long onCmdToggleHideConnections(FXObject*, FXSelector, void*);

    /// @brief toggle show additional sub-elements
    long onCmdToggleShowAdditionalSubElements(FXObject*, FXSelector, void*);

    /// @brief toggle show TAZ elements
    long onCmdToggleShowTAZElements(FXObject*, FXSelector, void*);

    /// @brief toggle extend selection
    long onCmdToggleExtendSelection(FXObject*, FXSelector, void*);

    /// @brief toggle change all phases
    long onCmdToggleChangeAllPhases(FXObject*, FXSelector, void*);

    /// @brief toggle show grid
    long onCmdToggleShowGrid(FXObject*, FXSelector, void*);

    /// @brief toggle draw junction shape
    long onCmdToggleDrawJunctionShape(FXObject*, FXSelector, void*);

    /// @brief toggle draw vehicles in begin position or spread in lane
    long onCmdToggleDrawSpreadVehicles(FXObject*, FXSelector, void*);

    /// @brief toggle warn for merge
    long onCmdToggleWarnAboutMerge(FXObject*, FXSelector, void*);

    /// @brief toggle show junction bubbles
    long onCmdToggleShowJunctionBubbles(FXObject*, FXSelector, void*);

    /// @brief toggle move elevation
    long onCmdToggleMoveElevation(FXObject*, FXSelector, void*);

    /// @brief toggle chain edges
    long onCmdToggleChainEdges(FXObject*, FXSelector, void*);

    /// @brief toggle autoOpposite edge
    long onCmdToggleAutoOppositeEdge(FXObject*, FXSelector, void*);

    /// @brief toggle hide non inspected demand elements
    long onCmdToggleHideNonInspecteDemandElements(FXObject*, FXSelector, void*);

    /// @brief toggle hide non inspected demand elements
    long onCmdToggleShowOverlappedRoutes(FXObject*, FXSelector, void*);

    /// @brief toggle hide shapes in super mode demand
    long onCmdToggleHideShapes(FXObject*, FXSelector, void*);

    /// @brief toggle show all trips in super mode demand
    long onCmdToggleShowTrips(FXObject*, FXSelector, void*);

    /// @brief toggle show all person plans in super mode demand
    long onCmdToggleShowAllPersonPlans(FXObject*, FXSelector, void*);

    /// @brief toggle lock person in super mode demand
    long onCmdToggleLockPerson(FXObject*, FXSelector, void*);

    /// @brief toggle show all container plans in super mode demand
    long onCmdToggleShowAllContainerPlans(FXObject*, FXSelector, void*);

    /// @brief toggle lock container in super mode demand
    long onCmdToggleLockContainer(FXObject*, FXSelector, void*);

    /// @brief toggle show additionals in super mode data
    long onCmdToggleShowAdditionals(FXObject*, FXSelector, void*);

    /// @brief toggle show shapes in super mode data
    long onCmdToggleShowShapes(FXObject*, FXSelector, void*);

    /// @brief toggle show demand elements (network)
    long onCmdToggleShowDemandElementsNetwork(FXObject*, FXSelector, void*);

    /// @brief toggle show demand elements (data)
    long onCmdToggleShowDemandElementsData(FXObject*, FXSelector, void*);

    /// @brief toggle TAZRel drawing
    long onCmdToggleTAZRelDrawing(FXObject*, FXSelector, void*);

    /// @brief toggle TAZdrawFill
    long onCmdToggleTAZDrawFill(FXObject*, FXSelector, void*);

    /// @brief toggle TAZRez only from
    long onCmdToggleTAZRelOnlyFrom(FXObject*, FXSelector, void*);

    /// @brief toggle TAZRez only to
    long onCmdToggleTAZRelOnlyTo(FXObject*, FXSelector, void*);

    /// @}

    //// @name interval bar functions
    /// @{

    /// @brief change generic data type in interval bar
    long onCmdIntervalBarGenericDataType(FXObject*, FXSelector, void*);

    /// @brief change data set in interval bar
    long onCmdIntervalBarDataSet(FXObject*, FXSelector, void*);

    /// @brief change limit interval in interval bar
    long onCmdIntervalBarLimit(FXObject*, FXSelector, void*);

    /// @brief change begin in interval bar
    long onCmdIntervalBarSetBegin(FXObject*, FXSelector, void*);

    /// @brief change end in interval bar
    long onCmdIntervalBarSetEnd(FXObject*, FXSelector, void*);

    /// @brief change parameter in interval bar
    long onCmdIntervalBarSetParameter(FXObject*, FXSelector, void*);

    /// @}

    /// @brief select AC under cursor
    long onCmdAddSelected(FXObject*, FXSelector, void*);

    /// @brief unselect AC under cursor
    long onCmdRemoveSelected(FXObject*, FXSelector, void*);

    /// @brief select Edge under cursor
    long onCmdAddEdgeSelected(FXObject*, FXSelector, void*);

    /// @brief unselect Edge under cursor
    long onCmdRemoveEdgeSelected(FXObject*, FXSelector, void*);

    /// @brief abort current edition operation
    void abortOperation(bool clearSelection = true);

    /// @brief handle del keypress
    void hotkeyDel();

    /// @brief handle enter keypress
    void hotkeyEnter();

    /// @brief handle backspace keypress
    void hotkeyBackSpace();

    /// @brief handle focus frame keypress
    void hotkeyFocusFrame();

    /// @brief get the net object
    GNEViewParent* getViewParent() const;

    /// @brief get the net object
    GNENet* getNet() const;

    /// @brief get the undoList object
    GNEUndoList* getUndoList() const;

    /// @brief get interval bar
    GNEViewNetHelper::IntervalBar& getIntervalBar();

    /// @brief get inspected attribute carriers
    const std::vector<GNEAttributeCarrier*>& getInspectedAttributeCarriers() const;

    /// @brief get lock manager
    GNEViewNetHelper::LockManager& getLockManager();

    /// @brief set inspected attributeCarrier
    void setInspectedAttributeCarriers(const std::vector<GNEAttributeCarrier*> ACs);

    /// @brief check if attribute carrier is being inspected
    bool isAttributeCarrierInspected(const GNEAttributeCarrier* AC) const;

    /// @brief remove given AC of list of inspected Attribute Carriers
    void removeFromAttributeCarrierInspected(const GNEAttributeCarrier* AC);

    /// @brief get front attributeCarrier
    const GNEAttributeCarrier* getFrontAttributeCarrier() const;

    /// @brief set front attributeCarrier
    void setFrontAttributeCarrier(GNEAttributeCarrier* AC);

    /// @brief draw front attributeCarrier
    void drawTranslateFrontAttributeCarrier(const GNEAttributeCarrier* AC, double typeOrLayer, const double extraOffset = 0);

    /// @brief get last created route
    GNEDemandElement *getLastCreatedRoute() const;

    /// @brief set last created route
    void setLastCreatedRoute(GNEDemandElement *lastCreatedRoute);

    /// @brief set statusBar text
    void setStatusBarText(const std::string& text);

    /// @brief whether to autoselect nodes or to lanes
    bool autoSelectNodes();

    /// @brief set selection scaling (in GNESelectorFrame)
    void setSelectorFrameScale(double selectionScale);

    /// @brief update control contents after undo/redo or recompute
    void updateControls();

    /// @brief change all phases
    bool changeAllPhases() const;

    /// @brief return true if junction must be showed as bubbles
    bool showJunctionAsBubbles() const;

    /// @brief try to merge moved junction with another junction in that spot return true if merging did take place
    bool mergeJunctions(GNEJunction* movedJunction, GNEJunction* targetJunction);

    /// @brief ask about change supermode
    bool aksChangeSupermode(const std::string& operation, Supermode expectedSupermode);

protected:
    /// @brief FOX needs this
    GNEViewNet();

    /// @brief do paintGL
    int doPaintGL(int mode, const Boundary& bound);

    /// @brief called after some features are already initialized
    void doInit();

private:
    /// @name structs related with modes and testing mode
    /// @{

    /// @brief variable used to save variables related with edit moves modes
    GNEViewNetHelper::EditModes myEditModes;

    /// @brief variable used to save variables related with testing mode
    GNEViewNetHelper::TestingMode myTestingMode;
    /// @}

    /// @name structs related with input (keyboard and mouse)
    /// @{

    /// @brief variable used to save key status after certain events
    GNEViewNetHelper::MouseButtonKeyPressed myMouseButtonKeyPressed;

    /// @brief variable use to save all pointers to objects under cursor after a click
    GNEViewNetHelper::ObjectsUnderCursor myObjectsUnderCursor;
    /// @}

    /// @name structs related with checkable buttons
    /// @{

    /// @brief variable used to save checkable buttons for common supermodes
    GNEViewNetHelper::CommonCheckableButtons myCommonCheckableButtons;

    /// @brief variable used to save checkable buttons for Supermode Network
    GNEViewNetHelper::NetworkCheckableButtons myNetworkCheckableButtons;

    /// @brief variable used to save checkable buttons for Supermode Demand
    GNEViewNetHelper::DemandCheckableButtons myDemandCheckableButtons;

    /// @brief variable used to save checkable buttons for Supermode Data
    GNEViewNetHelper::DataCheckableButtons myDataCheckableButtons;
    /// @}

    /// @name structs related with view options
    /// @{

    /// @brief variable used to save variables related with view options in supermode Network
    GNEViewNetHelper::NetworkViewOptions myNetworkViewOptions;

    /// @brief variable used to save variables related with view options in supermode Demand
    GNEViewNetHelper::DemandViewOptions myDemandViewOptions;

    /// @brief variable used to save variables related with view options in supermode Data
    GNEViewNetHelper::DataViewOptions myDataViewOptions;
    /// @}

    /// @brief variable used to save IntervalBar
    GNEViewNetHelper::IntervalBar myIntervalBar;

    /// @name structs related with move elements
    /// @{
    /// @brief variable used to save variables related with movement of single elements
    GNEViewNetHelper::MoveSingleElementValues myMoveSingleElementValues;

    /// @brief variable used to save variables related with movement of multiple elements
    GNEViewNetHelper::MoveMultipleElementValues myMoveMultipleElementValues;
    // @}

    /// @name structs related with Demand options
    /// @{

    /// @brief variable used to save variables related with vehicle options
    GNEViewNetHelper::VehicleOptions myVehicleOptions;

    /// @brief variable used to save variables related with vehicle type options
    GNEViewNetHelper::VehicleTypeOptions myVehicleTypeOptions;
    // @}

    /// @brief variable used to save elements
    GNEViewNetHelper::SaveElements mySaveElements;

    /// @brief variable used to save variables related with selecting areas
    GNEViewNetHelper::SelectingArea mySelectingArea;

    /// @brief struct for grouping all variables related with edit shapes
    GNEViewNetHelper::EditNetworkElementShapes myEditNetworkElementShapes;

    /// @brief lock manager
    GNEViewNetHelper::LockManager myLockManager;

    /// @brief view parent
    GNEViewParent* myViewParent = nullptr;

    /// @brief Pointer to current net. (We are not responsible for deletion)
    GNENet* myNet = nullptr;

    /// @brief the current frame
    GNEFrame* myCurrentFrame = nullptr;

    /// @brief a reference to the undolist maintained in the application
    GNEUndoList* myUndoList = nullptr;

    /// @brief current inspected attribute carrier
    std::vector<GNEAttributeCarrier*> myInspectedAttributeCarriers;

    /// @brief front attribute carrier
    GNEAttributeCarrier* myFrontAttributeCarrier = nullptr;

    /// @brief last created route
    GNEDemandElement* myLastCreatedRoute = nullptr;

    /// @brief create edit mode buttons and elements
    void buildEditModeControls();

    /// @brief updates Network mode specific controls
    void updateNetworkModeSpecificControls();

    /// @brief updates Demand mode specific controls
    void updateDemandModeSpecificControls();

    /// @brief updates Data mode specific controls
    void updateDataModeSpecificControls();

    /// @brief delete given network attribute carriers
    void deleteNetworkAttributeCarriers(const std::vector<GNEAttributeCarrier*> ACs);

    /// @brief delete given demand attribute carriers
    void deleteDemandAttributeCarriers(const std::vector<GNEAttributeCarrier*> ACs);

    /// @brief delete data attribute carriers
    void deleteDataAttributeCarriers(const std::vector<GNEAttributeCarrier*> ACs);

    /// @brief try to retrieve an edge at popup position
    GNEEdge* getEdgeAtPopupPosition();

    /// @brief try to retrieve a lane at popup position
    GNELane* getLaneAtPopupPosition();

    /// @brief try to retrieve a junction at popup position
    GNEJunction* getJunctionAtPopupPosition();

    /// @brief try to retrieve a connection at popup position
    GNEConnection* getConnectionAtPopupPosition();

    /// @brief try to retrieve a crossing at popup position
    GNECrossing* getCrossingAtPopupPosition();

    /// @brief try to retrieve a additional at popup position
    GNEAdditional* getAdditionalAtPopupPosition();

    /// @brief try to retrieve a polygon at popup position
    GNEPoly* getPolygonAtPopupPosition();

    /// @brief try to retrieve a POILane at popup position
    GNEPOI* getPOIAtPopupPosition();

    /// @brief try to retrieve a TAZ at popup position
    GNETAZ* getTAZAtPopupPosition();

    /// @brief restrict lane
    bool restrictLane(SUMOVehicleClass vclass);

    /// @brief add restricted lane
    bool addRestrictedLane(SUMOVehicleClass vclass, const bool insertAtFront);

    /// @brief remove restricted lane
    bool removeRestrictedLane(SUMOVehicleClass vclass);

    /// @brief Auxiliar function used by onLeftBtnPress(...)
    void processClick(void* eventData);

    /// @brief update cursor after every click/key press/release
    void updateCursor();

    /// @brief draw functions
    /// @{

    /// @brief draw temporal polygon shape in Polygon Mode
    void drawTemporalDrawingShape() const;

    /// @brief draw temporal junction in create edge mode
    void drawTemporalJunction() const;
    /// @}

    /// @brief mouse process functions
    /// @{

    /// @brief process left button press function in Supermode Network
    void processLeftButtonPressNetwork(void* eventData);

    /// @brief process left button release function in Supermode Network
    void processLeftButtonReleaseNetwork();

    /// @brief process move mouse function in Supermode Network
    void processMoveMouseNetwork(const bool mouseLeftButtonPressed);

    /// @brief process left button press function in Supermode Demand
    void processLeftButtonPressDemand(void* eventData);

    /// @brief process left button release function in Supermode Demand
    void processLeftButtonReleaseDemand();

    /// @brief process move mouse function in Supermode Demand
    void processMoveMouseDemand(const bool mouseLeftButtonPressed);

    /// @brief process left button press function in Supermode Data
    void processLeftButtonPressData(void* eventData);

    /// @brief process left button release function in Supermode Data
    void processLeftButtonReleaseData();

    /// @brief process move mouse function in Supermode Data
    void processMoveMouseData(const bool mouseLeftButtonPressed);

    /// @brief Invalidated copy constructor.
    GNEViewNet(const GNEViewNet&) = delete;

    /// @brief Invalidated assignment operator.
    GNEViewNet& operator=(const GNEViewNet&) = delete;
};
