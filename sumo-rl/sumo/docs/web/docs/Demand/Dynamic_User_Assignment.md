---
title: Dynamic User Assignment
---

# Introduction

For a given set of vehicles with of origin-destination relations
(trips), the simulation must determine routes through the network (list
of edges) that are used to reach the destination from the origin edge.
The simplest method to find these routes is by computing shortest or
fastest routes through the network using a routing algorithm such as
Dijkstra or A\*. These algorithms require assumptions regarding the
travel time for each network edge which is commonly not known before
running the simulation due to the fact that travel times depend on the
number of vehicles in the network.

!!! caution
    A frequent problem with naive user assignment is that all vehicles take the fastest path under the assumption that they are alone in the network and are then jammed at bottlenecks due to the sheer amount of traffic.

The problem of determining suitable routes that take into account travel
times in a traffic-loaded network is called *user assignment*. SUMO
provides different tools to solve this problem and they are described
below.

# Iterative Assignment (**D**ynamic **U**ser **E**quilibrium)

The tool [duaIterate.py](../Tools/Assign.md#duaiteratepy) can be used to compute the
(approximate) dynamic user equilibrium.

!!! caution
    This script will require copious amounts of disk space

```
python tools/assign/duaIterate.py -n <network-file> -t <trip-file> -l <nr-of-iterations>
```

*duaIterate.py* supports many of the same options as
[sumo](../sumo.md). Any options not listed when calling
*duaIterate.py* ***--help*** can be passed to [sumo](../sumo.md) by adding **sumo--long-option-name arg**
after the regular options (i.e. **sumo--step-length 0.5**). The same is true for duarouter options
using **duarouter--long-option-name arg**. Be aware that those options have to come *after* the regular
options.

This script tries to calculate a user equilibrium, that is, it tries to
find a route for each vehicle (each trip from the trip-file above) such
that each vehicle cannot reduce its travel cost (usually the travel
time) by using a different route. It does so iteratively (hence the
name) by

1.  calling [duarouter](../duarouter.md) to route the vehicles in a
    network with the last known edge costs (starting with empty-network
    travel times)
2.  calling [sumo](../sumo.md) to simulate "real" travel times
    result from the calculated routes. The result edge costs are used in
    the net routing step.

The number of iterations may be set to a fixed number of determined
dynamically depending on the used options. In order to ensure
convergence there are different methods employed to calculate the route
choice probability from the route cost (so the vehicle does not always
choose the "cheapest" route). In general, new routes will be added by
the router to the route set of each vehicle in each iteration (at least
if none of the present routes is the "cheapest") and may be chosen
according to the route choice mechanisms described below.

Between successive calls of duarouter, the *.rou.alt.xml* format is used
to record not only the current *best* route but also previously computed
alternative routes. These routes are collected within a route
distribution and used when deciding the actual route to drive in the
next simulation step. This isn't always the one with the currently
lowest cost but is rather sampled from the distribution of alternative
routes by a configurable algorithm described below.

## Route-Choice algorithm

The two methods which are implemented are called
[Gawron](../Publications.md#traffic_assignment) and
[Logit](https://en.wikipedia.org/wiki/Discrete_choice) in the following.
The input for each of the methods is a weight or cost function \(w\) on
the edges of the net, coming from the simulation or default costs (in
the first step or for edges which have not been traveled yet), and a set
of routes <img src="http://latex.codecogs.com/gif.latex?R" border="0" style="margin:0;"/> where each route <img src="http://latex.codecogs.com/gif.latex?r" border="0" style="margin:0;"/> has an old cost <img src="http://latex.codecogs.com/gif.latex?c_r" border="0" style="margin:0;"/> and an
old probability <img src="http://latex.codecogs.com/gif.latex?p_r" border="0" style="margin:0;"/> (from the last iteration) and needs a new cost
<img src="http://latex.codecogs.com/gif.latex?c_r'" border="0" style="margin:0;"/> and a new probability <img src="http://latex.codecogs.com/gif.latex?p_r'" border="0" style="margin:0;"/>.

### Gawron (default)

The Gawron algorithm computes probabilities for choosing from a set of
alternative routes for each driver. The following values are considered
to compute these probabilities:

- the travel time along the used route in the previous simulation step
- the sum of edge travel times for a set of alternative routes
- the previous probability of choosing a route

#### Number of Routes in each traveller's route set

The maximum number of routes can be defined by users, where 5 is the default value. In each iteration, the route usage probability is calculated for each route. When the number of routes is larger than the defined amount, routes with smallest probabilities are removed.

#### Updates of Travel Time

The update rule is explained with the following example. Driver d chooses Route r in Iteration i. The travel time Tau_d(r, i+1) is calculated according to the aggregated and averaged link travel times per defined interval (default: 900 s) in Iteration i. The travel time for Driver d's Route r in Iteration i+1 equals to Tau_d(r, i) as indicated in Formula (1). The travel times of the other routes in Driver d's route set are then updated with Formula (2) respectively, where Tau_d(s, i) is the travel time needed to travel on Route s in Iteration i and calculated with the same way used for calculating Tau_d(r, i) an T_d(s, i-1). The parameter beta is to prevent travellers from strongly "remembering" the latest trave time of each route in their route sets. The current default value for beta is 0.3.
 
T_d(r, i+1) = Tau_d(r, i) ------------------------------------(1)
  
T_d(s, i+1) = beta * Tau_d(s, i) + (1 - beta) * T_d(s, i-1) ---(2)

, where s is one of the routes, which are not selected to use in Iteration i, in Driver d's route set.

The aforementioned update rules also apply when other travel cost units are used. The way to use simulated link costs for calcuating route costs may result in cost underestimation especially when significant congestion only on one of traffic movenents (e.g. left-turn or right-turn) exists. The existing ticket #2566 deals with this issue. In Formula (1), it is also possible to use Driver d's actual travel cost in Iteration i as Tau_d(r, i).

### Logit

The Logit mechanism applies a fixed formula to each route to calculate
the new probability. It ignores old costs and old probabilities and
takes the route cost directly as the sum of the edge costs from the last
simulation.

<img src="http://latex.codecogs.com/gif.latex?c_r' = \sum_{e\in r}w(e)" border="0" style="margin:0;"/>

The probabilities are calculated from an exponential function with
parameter <img src="http://latex.codecogs.com/gif.latex?\theta" border="0" style="margin:0;"/> scaled by the sum over all route values:

<img src="http://latex.codecogs.com/gif.latex?p_r' = \frac{\exp(\theta c_r')}{\sum_{s\in R}\exp(\theta c_s')}" border="0" style="margin:0;"/>

!!! caution
    It is recommended to set option **--convergence-steps** (i.e. to the same number as **-last-step**) to ensure convergence. Otherwise Logit route choice may keep oscillating, especially with higher values of **--logittheta**.

## Termination

DuaIterate convergence is hard to predict and results may continue to vary even after 1000 iterations.
There are several strategies in this regard:

### Default

By default, a fixed number of iterations, configured via **--first-step** and **--last-step** (default 50) is performed.

### Deviation in Average Travel times

The option **--max-convergence-deviation** may be used to detect convergence and abort iterations
automatically. In each iteration, the average travel time of all trips is computed. From the sequence of these values (one per iteration), the relative standard deviation is computed. Onece a minimum number of iterations has been computed (**--convergence-iterations**, default 10) and this deviation falls below the max-convergence deviation threshold, iterations are aborted 

### Forced convergence

Option **--convergence-steps** may used to force convergence by iteratively reducing the fraction of vehicles that may alter their route.

- If a positive value x is used, the fraction of vehicles that keep their old route is set to `max(0, min(step / x, 1)` which prevents changes in assignment after step x.
- If a negative value x is used, the fraction of vehicles that keep their old route is set to `1 - 1.0 / (step - |x|)` for steps after `|x|` which asymptotically reduces assignment after `|x|` steps.

## Speeding up Iterations

There is currently now way to speed up duaIteraty.py by parallelization.
However, the total running time of duaIterate is strongly influenced by the total running time of "jammed" iterations.
This is a frequent occurrence in the early iterations where many cars try to take the fastest route while disregarding capacity.
There are several options to mitigate this:

- by ramping up the traffic scaling so the first iterations have fewer traffic (**--inc-start, --inc-base, --inc-max, --incrementation**)
- by aborting earlier iterations at an earlier time (**--time-inc**)
- by giving the initial demand with a sensible starting solution (i.e. computed by marouter) along with option **--skip-first-routing**
- by trying to carry more information between runs (**--weight-memory, --pessimism**)

## Usage Examples

### Loading vehicle types from an additional file

By default, vehicle types are taken from the input trip file and are
then propagated through [duarouter](../duarouter.md) iterations
(always as part of the written route file).

In order to use vehicle type definitions from an {{AdditionalFile}}, further options must
be set

```
duaIterate.py -n ... -t ... -l ... 
  --additional-file <FILE_WITH_VTYPES> 
  duarouter--aditional-file <FILE_WITH_VTYPES> 
  duarouter--vtype-output dummy.xml
```

Options preceded by the string *duarouter--* are passed directly to
duarouter and the option *vtype-output dummy.xml* must be used to
prevent duplicate definition of vehicle types in the generated output
files.

# oneShot-assignment

An alternative to the iterative user assignment above is incremental
assignment. This happens automatically when using `<trip>` input directly in
[sumo](../sumo.md) instead of `<vehicle>`s with pre-defined routes. In this
case each vehicle will compute a fastest-path computation at the time of
departure which prevents all vehicles from driving blindly into the same
jam and works pretty well empirically (for larger scenarios).

The routes for this incremental assignment are computed using the
[Automatic Routing / Routing Device](../Demand/Automatic_Routing.md) mechanism. 
It is also possible to enable periodic rerouting to allow increased reactivity to developing jams.

Since automatic rerouting allows for various configuration options, the script
[Tools/Assign\#one-shot.py](../Tools/Assign.md#one-shotpy) may be
used to automatically try different parameter settings.

# [marouter](../marouter.md)

The [marouter](../marouter.md) application computes a *classic*
macroscopic assignment. It employs mathematical functions (resistive
functions) that approximate travel time increases when increasing flow.
This allows to compute an iterative assignment without the need for
time-consuming microscopic simulation.
