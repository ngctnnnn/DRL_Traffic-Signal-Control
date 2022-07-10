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
/// @file    SUMOTrafficObject.h
/// @author  Jakob Erdmann
/// @date    Mon, 25 Mar 2019
///
// Abstract base class for vehicle and person representations
/****************************************************************************/
#pragma once
#include <config.h>

#include <vector>
#include <typeinfo>
#include <utils/common/SUMOTime.h>
#include <utils/common/Named.h>
#include <utils/common/SUMOVehicleClass.h>


// ===========================================================================
// class declarations
// ===========================================================================
class MSVehicleType;
class SUMOVehicleParameter;
class MSEdge;
class MSLane;
class Position;

// ===========================================================================
// class definitions
// ===========================================================================
/**
 * @class SUMOTrafficObject
 * @brief Representation of a vehicle, person, or container
 */
class SUMOTrafficObject : public Named {
public:

    /// @brief Constructor
    SUMOTrafficObject(const std::string& id) : Named(id) {}

    /// @brief Destructor
    virtual ~SUMOTrafficObject() {}

    /** @brief Whether it is a vehicle
     * @return true for vehicles, false otherwise
     */
    virtual bool isVehicle() const {
        return false;
    }

    /** @brief Whether it is a person
     * @return true for persons, false otherwise
     */
    virtual bool isPerson() const {
        return false;
    }

    /** @brief Whether it is a container
     * @return true for containers, false otherwise
     */
    virtual bool isContainer() const {
        return false;
    }

    /** @brief Returns the object's "vehicle" type
     * @return The vehicle's type
     */
    virtual const MSVehicleType& getVehicleType() const = 0;

    /** @brief Replaces the current vehicle type by the one given
    *
    * If the currently used vehicle type is marked as being used by this object
    *  only, it is deleted, first. The new, given type is then assigned to
    *  "myVType".
    * @param[in] type The new vehicle type
    * @see MSTransportable::myVType
    */
    virtual void replaceVehicleType(MSVehicleType* type) = 0;


    /** @brief Returns the vehicle's parameter (including departure definition)
     *
     * @return The vehicle's parameter
     */
    virtual const SUMOVehicleParameter& getParameter() const = 0;

    /** @brief Returns the associated RNG for this object
    * @return The vehicle's associated RNG
    */
    virtual SumoRNG* getRNG() const = 0;

    /** @brief Returns whether the object is at a stop
     * @return Whether it has stopped
     */
    virtual bool isStopped() const = 0;

    /** @brief Returns the edge the object is currently at
     *
     * @return The current edge in the object's route
     */
    virtual const MSEdge* getEdge() const = 0;

    /** @brief Returns the lane the object is currently at
     *
     * @return The current lane or nullptr if the object is not on a lane
     */
    virtual const MSLane* getLane() const = 0;

    /** @brief Returns the slope of the road at object's position in degrees
     * @return The slope
     */
    virtual double getSlope() const = 0;

    virtual double getChosenSpeedFactor() const = 0;

    /** @brief Returns the object's access class
     * @return The object's access class
     */
    virtual SUMOVehicleClass getVClass() const = 0;

    /** @brief Returns the object's maximum speed
     * @return The object's maximum speed
     */
    virtual double getMaxSpeed() const = 0;

    virtual SUMOTime getWaitingTime() const = 0;

    /** @brief Returns the object's current speed
     * @return The object's speed
     */
    virtual double getSpeed() const = 0;

    // This definition was introduced to make the MSVehicle's previousSpeed Refs. #2579
    /** @brief Returns the object's previous speed
     * @return The object's previous speed
     */
    virtual double getPreviousSpeed() const = 0;


    /** @brief Returns the object's acceleration
     * @return The acceleration
     */
    virtual double getAcceleration() const = 0;

    /** @brief Get the object's position along the lane
     * @return The position of the object (in m from the lane's begin)
     */
    virtual double getPositionOnLane() const = 0;

    /** @brief Get the object's back position along the given lane
     * @return The position of the object (in m from the given lane's begin)
     */
    virtual double getBackPositionOnLane(const MSLane* lane) const = 0;


    /** @brief Return current position (x/y, cartesian)
     *
     * If the object is not in the net, Position::INVALID.
     * @param[in] offset optional offset in longitudinal direction
     * @return The current position (in cartesian coordinates)
     * @see myLane
     */
    virtual Position getPosition(const double offset = 0) const = 0;

    /** @brief Returns the object's angle in degrees
     */
    virtual double getAngle() const = 0;

    /** @brief Returns whether this object has arrived
     */
    virtual bool hasArrived() const = 0;

    /// @brief whether the vehicle is individually influenced (via TraCI or special parameters)
    virtual bool hasInfluencer() const = 0;

    /// @brief whether this object is selected in the GUI
    virtual bool isSelected() const = 0;


};
