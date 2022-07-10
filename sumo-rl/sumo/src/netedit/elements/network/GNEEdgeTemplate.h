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
/// @file    GNEEdgeTemplate.h
/// @author  Pablo Alvarez Lopez
/// @date    Dec 2021
///
// Template for edges
/****************************************************************************/
#pragma once
#include <config.h>

#include <netedit/elements/GNEAttributeCarrier.h>

// ===========================================================================
// class declarations
// ===========================================================================
class GNEEdge;
class GNELaneTemplate;

// ===========================================================================
// class definitions
// ===========================================================================
class GNEEdgeTemplate : public GNEAttributeCarrier {

public:
    /// @brief Constructor
    GNEEdgeTemplate(const GNEEdge* edge);

    /// @brief Destructor.
    ~GNEEdgeTemplate();

    /// @brief get GNEHierarchicalElement associated with this AttributeCarrier
    GNEHierarchicalElement* getHierarchicalElement();

    /// @brief get vector with the lane templates of this edge
    const std::vector<GNELaneTemplate*>& getLaneTemplates() const;

    /// @brief update lane templates
    void updateLaneTemplates();

    /// @name Function related with graphics
    /// @{
    /// @brief get ID (all Attribute Carriers have one)
    const std::string& getID() const;

    /// @brief get GUIGlObject associated with this AttributeCarrier
    GUIGlObject* getGUIGlObject();

    /// @brief update pre-computed geometry information
    void updateGeometry();

    /// @}

    /// @name Functions related with attributes
    /// @{
    /* @brief method for getting the Attribute of an XML key
     * @param[in] key The attribute key
     * @return string with the value associated to key
     */
    std::string getAttribute(SumoXMLAttr key) const;

    /* @brief method for setting the attribute and letting the object perform additional changes
     * @param[in] key The attribute key
     * @param[in] value The new value
     * @param[in] undoList The undoList on which to register changes
     */
    void setAttribute(SumoXMLAttr key, const std::string& value, GNEUndoList* undoList);

    /* @brief method for check if new value for certain attribute is valid
     * @param[in] key The attribute key
     * @param[in] value The new value
     */
    bool isValid(SumoXMLAttr key, const std::string& value);

    /* @brief method for enable attribute
     * @param[in] key The attribute key
     * @param[in] undoList The undoList on which to register changes
     * @note certain attributes can be only enabled, and can produce the disabling of other attributes
     */
    void enableAttribute(SumoXMLAttr key, GNEUndoList* undoList);

    /* @brief method for disable attribute
     * @param[in] key The attribute key
     * @param[in] undoList The undoList on which to register changes
     * @note certain attributes can be only enabled, and can produce the disabling of other attributes
     */
    void disableAttribute(SumoXMLAttr key, GNEUndoList* undoList);

    /* @brief method for check if the value for certain attribute is set
     * @param[in] key The attribute key
     */
    bool isAttributeEnabled(SumoXMLAttr key) const;

    /* @brief method for check if the value for certain attribute is computed (for example, due a network recomputing)
     * @param[in] key The attribute key
     */
    bool isAttributeComputed(SumoXMLAttr key) const;

    /// @brief get PopPup ID (Used in AC Hierarchy)
    std::string getPopUpID() const;

    /// @brief get Hierarchy Name (Used in AC Hierarchy)
    std::string getHierarchyName() const;

    /// @}

    /// @brief get parameters map
    const Parameterised::Map& getACParametersMap() const;

protected:
    /// @brief pointer to original edge
    const GNEEdge* myEdge;

    /// @brief vector with the lane templates of this edge
    std::vector<GNELaneTemplate*> myLaneTemplates;

private:
    /// @brief set attribute after validation
    void setAttribute(SumoXMLAttr key, const std::string& value);

    /// @brief method for enable or disable the attribute and nothing else (used in GNEChange_EnableAttribute)
    void toogleAttribute(SumoXMLAttr key, const bool value);

    /// @brief invalidated copy constructor
    GNEEdgeTemplate(const GNEEdgeTemplate& s) = delete;

    /// @brief invalidated assignment operator
    GNEEdgeTemplate& operator=(const GNEEdgeTemplate& s) = delete;
};
