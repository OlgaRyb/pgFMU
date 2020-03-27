1. _fmu_create (modelRef, [instanceId]) -> instanceId_

This function creates an in-DBMS instance of an FMU model. It takes the path to the model as a mandatory argument, and an ID(name) for the instance as an optional argument. It returns the ID of the FMU model instance in the database.

Usage example:

```sql
SELECT fmu_create ('/tmp/hp1.fmu', 'HP1Instance1');
```

2. _fmu_copy (instanceId, [instanceId2]) -> instanceId2_

This function makes a copy of the existing model instance. As a mandatory argument, it takes the instanceID of the ine that is supposed to be copied.

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

5. fmu_set_initial (instanceId, varName, initialValue) -> instanceId
Usage example:
SELECT fmu_set_initial ('HP1Instance1', 'A', 0)

6. fmu_set_minimum (instanceId, varName, initialValue) -> instanceId
Usage example:
SELECT fmu_set_minimum ('HP1Instance1', 'A', -20)

7. fmu_set_maximum (instanceId, varName, initialValue) -> instanceId
Usage example:
SELECT fmu_set_maximum ('HP1Instance1', 'A', 20)

8. fmu_reset (instanceId) -> instanceId
Usage example:
SELECT fmu_reset ('HP1instance1')

9. fmu_delete_model (instanceId)
Usage example:
SELECT fmu_reset ('265ijhd83hdkas8')

10. fmu_delete_instance (instanceId)
Usage example:
SELECT fmu_reset ('HP1Instance1')

11. fmu_parest (instanceIds, input_sqls, [pars], [threshold]) -> estimationErrors
Usage example:
SELECT fmu_parest ('{HP1Instance1, HP1Instance2}', '{SELECT ' FROM measurements, SELECT ' FROM measurements2}', '{A, B}')

12. fmu_simulate (instanceId, [input_sql], [time_to], [time_from]) -> (simulaitonTime, instanceId, values)
Usage example:
SELECT simulationTime, instanceId, value FROM fmu_simulate ('HP1Instance1', 'SELECT * FROM measurements') WHERE varName IN ('y', 'x')
