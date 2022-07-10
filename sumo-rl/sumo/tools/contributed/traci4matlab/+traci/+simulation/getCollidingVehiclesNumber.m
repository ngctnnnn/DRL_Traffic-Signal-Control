function collidingNumber = getCollidingVehiclesNumber()
%getCollidingVehiclesNumber
%   collidingNumber = getCollidingVehiclesNumber()

%   Copyright 2019 Universidad Nacional de Colombia,
%   Politecnico Jaime Isaza Cadavid.
%   Authors: Andres Acosta, Jairo Espinosa, Jorge Espinosa.
%   $Id: getCollidingVehiclesNumber.m 54 2019-01-03 15:41:54Z afacostag $

import traci.constants
collidingNumber = traci.simulation.getUniversal(constants.VAR_COLLIDING_VEHICLES_NUMBER);
