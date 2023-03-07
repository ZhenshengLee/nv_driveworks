# Sample Schedule Manager App

This application can be used as a reference for implementing the schedule manager for different client applications. It can be configured to perform different schedule switch scenarios. It can go through each schedule in the following way.
- choose a schedule
- starts executing the schedule
- runs the schedule for provided interval
- stops the schedule
- switches to the next schedule

Currently, this sample cycles between 2 fixed schedule ids 101 and 102. (Modification needed for general use case)


## Schedule Manager Usage:

Schedule Manager takes the following command line parameters: 
- Mandatory Arguments
  - The name of the Schedule Manager which needs to connect to the STM Master (The same name must be given to the STM Master as an input)
- Optional Arguments
  - -v : verbose
  - -p : Specify the period (ms) for which schedules are executed
  - -g : Specify the break/delta (ms) between the end of the previous execution and start of the next execution
  - -r : Maximum schedule index to reach before resetting
  - -c : Number of times the execution cycle is to be repeated
  - -x : allow exceptions during stopping and starting the schedule
[start up STM Master](../../runtime/README.md), and then simply use:

```bash
./stm_sample_manager <schedule manager name> ....
```