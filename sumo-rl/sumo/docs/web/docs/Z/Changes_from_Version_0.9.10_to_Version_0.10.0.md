---
title: Changes from Version 0.9.10 to Version 0.10.0
---

**Release date: 09.01.2009**

- Build
  - removed obsolete configuration option **--enable-speedcheck**

- All
  - using a reduced, faster XML parser (not validating)
  - default vehicle length is set to 7.5m (including gap)
  - route format changed ("edges" attribute)
  - (junction) internal lanes are the default now
  - copyright changed (removed obsolete ZAIK reference)

- sumo-gui
  - solved bug 2163422: Simulation does not start automatically
  - refactored the visualization speed-up; instead of using a
    self-made hack that uses a grid, an rtree structure is now used.
    This should solve problems with disappearing edges and nodes and
    with a slow selection
  - Added the possibility to save and load decals (int an XML-file)
  - View settings are now saved loaded using XML
  - implemented feature 1906106: make lines of bus stops always visible
  - implemented feature 2103556: Remove "Add Successors To Selected"

- Simulation
  - Refactored mean data; instead of having it as an additional item
    that has to be touched by vehicles, it is a MSMoveReminder
  - changed the definition of triggers; the old definition is still
    possible, but prints a warning.
    - <trigger objecttype="lane" attr="speed" objectid="<LANE\>\[;<LANE\>\]\*" ... --\> <variableSpeedSign lanes="<LANE\>\[;<LANE\>\]\*" ...
    - <trigger objecttype="rerouter" objectid="<EDGE\>\[;<EDGE\>\]\*" ... --\> <rerouter edges="<EDGE\>\[;<EDGE\>\]\*" ...
    - <trigger objecttype="emitter" objectid="<LANE\>" ... --\> <emitter lane="<LANE\>" ...
    - <trigger objecttype="bus_stop" objectid="<LANE\>" ... --\> <busStop lane="<LANE\>" ...
  - new "route probe" output added
  - solved bug 2350888: vehicle on false lane with cyclic routes
    (thanks to Daniel Janusz for supplying the example)
  - solved bug 2082959: bus stops that end at edge end confuse vehicles
    (thanks to Gerrit Lammert for supplying the example)
  - solved bug 2468327: log files together with input errors let sumo crash
  - solved bug 2227272: Random routes do not work with simple net
  - distributions for routes and vehicle types
  - changed format of dump definitions from command line to xml
    configuration

- netconvert
  - solved bug 2137657: edges disappear in combination with --keep-edges
  - changed APIs for setting lane-2-lane connections
  - work on Vissim-import
  - Now, the node positions are not added to an edge's geometry if
    one exists (if the edge has no explicite geometry, the node
    positions are still used). **--add-node-positions** was added,
    so that the old behavior is still available
  - Changed features of **--plain-outoput**: plain-nodes now contain
    tls information, additionally, plain-connections are written
  - changed computation whether a link is a left-mover
  - changed traffic lights computation
  - removed unneeded option **--all-logics**
  - removed "FileErrorReporter" usage
  - too complicated junctions (\#links\>64) are catched and set to
    unregulated
  - corrected computation of right-of-way rules
  - solved bug 2392943: netconvert fails when removing edges and guessing ramps
  - solved bug 2171355: turnarounds even with --no-turnarounds
  - building turnarounds may be now skipped for tls-controlled edges
    (using **--no-tls-turnarounds**)

- NETGEN
  - removed unneeded option **--all-logics**

- Router
  - solved bug 2149069: dfrouter does not work with internal lanes

- dfrouter
  - removed support for elmar's detector descriptions
  - removed **--fast-flows** option
  - solved bug 2165708: False speed conversion in dfrouter?
  - named classes properly
  - applied changes to trigger definitions (see "Simulation")
  - solved bug 2165666: dfrouter does not regard ''--all-end-follower"

- polyconvert
  - refactoring
  - in-line documentation added
  - added OSM-import functionality
  - added shape-file - import functionality (imports points, line
    strings, polygons, and their multi-pendants)
    - import is done via **--shape-files**
    - the id-field is given via **--shape-file.id-name *<NAME\>* **
    - optional projection guess is done via
      **--arcview.guess-projection**
  - renamed **--visum-file** to **--visum-files**; debugged