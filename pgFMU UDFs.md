1. _fmu_create (modelRef, [instanceId]) -> instanceId_

This function creates an in-DBMS instance of an FMU model. It takes the path to the model as a mandatory argument, and an ID(name) for the instance as an optional argument. It returns the ID of the FMU model instance in the database.

Usage example:

```sql
SELECT fmu_create ('/tmp/hp1.fmu', 'HP1Instance1');
```

2. _fmu_copy (instanceId, [instanceId2]) -> instanceId2_

This function makes a copy of the existing model instance. As a mandatory argument, it takes the instanceID of the one that is supposed to be copied.

Usage example:

```sql
SELECT fmu_copy ('HP1Instance1', 'HP1Instance2');
```

3. _fmu_get (instanceId, varName) -> (initialValue, minValue, maxValue)_

this function returns the initial, minimum, and maximum values of a specific variable.

Usage example:

```sql
SELECT * FROM fmu_get ('HP1Instance1', 'A')
```


4. _fmu_variables (instanceId) -> (instanceId, varName, varType, initialValue, minValue, maxValue)_

This funciton returns the information about the variables of a specific model instance.

Usage example:

```sql
SELECT * FROM fmu_variables ('HP1Instance1') AS f WHERE f.varType = 'parameter'
```

5. _fmu_set_initial (instanceId, varName, initialValue) -> instanceId_

This function sets the initial value of the specific model instance variable.

Usage example:


```sql
SELECT fmu_set_initial ('HP1Instance1', 'A', 0)
```

6. _fmu_set_minimum (instanceId, varName, initialValue) -> instanceId_

This function sets the minimum value of the specific model instance variable.

Usage example:

```sql
SELECT fmu_set_minimum ('HP1Instance1', 'A', -20)
```

7. _fmu_set_maximum (instanceId, varName, initialValue) -> instanceId_

This function sets the maximum value of the specific model instance variable.

Usage example:

```sql
SELECT fmu_set_maximum ('HP1Instance1', 'A', 20)
```

8. _fmu_delete_model (instanceId)_

This function delets an FMU model (the in-memory FMU model file) and all the "child" instances of this model.

Usage example:

```sql
SELECT fmu_reset ('265ijhd83hdkas8')
```

9. _fmu_delete_instance (instanceId)_

This function removes the specific model instance from the model catalogue (but not the model itself!) 

Usage example:

```sql
SELECT fmu_reset ('HP1Instance1')
```

10. _fmu_parest (instanceIds, input_sqls, variable, [pars], [threshold]) -> estimationErrors_

This function estimates the parameters of the FMU model instance. As input, it takes a list of instance IDs, a list of queries to retrieve the measurements data associated with the specific model instance, and a state variable of the model. As an optional parameter, the user can specify a list of parameters that are to be estimated, and the threshold (to which degree the measurements can be different, from 0 to 1).

Usage example:

```sql
SELECT fmu_parest ('{HP1Instance1, HP1Instance2}', '{SELECT ' FROM measurements, SELECT ' FROM measurements2}', 't', '{A, B}')
```

11. _fmu_simulate (instanceId, [input_sql], [time_to], [time_from]) -> (simulaitonTime, instanceId, values)_

This function simulates the FMU model instance. As input, it requires the instance ID as a mandatory argument. Optionally, the user can supply the table with the measurements, and the time range for simulation.

Usage example:

```sql
SELECT simulationTime, instanceId, value FROM fmu_simulate ('HP1Instance1', 'SELECT * FROM measurements') WHERE varName IN ('y', 'x')
```

