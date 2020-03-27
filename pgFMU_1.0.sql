-- Table to store FMU models

drop table if exists model cascade;
create table model (
	modelid			uuid primary key,
	name			text,
	description		text,
	defaultstarttime	numeric,
	defaultendtime		numeric,
	modelref		text
);

-- Table to store FMU model variables

drop type if exists fmu_vartype cascade;
create type fmu_vartype as enum(
			'state',
			'input',
			'output',
			'parameter'
);

drop table if exists modelvariable cascade;
create table modelvariable (
	modelid		uuid REFERENCES model(modelid) ON DELETE CASCADE,
	varname		text,
	vartype		fmu_vartype,
	initialvalue	numeric,
	minvalue	numeric,
	maxvalue	numeric,
	PRIMARY KEY (modelid, varname),
	FOREIGN KEY (modelid) REFERENCES model(modelid)
);

-- TODO:
-- impelment variant type for initialvalue, minvalue, maxvalue

-- Table to store FMU model instances
drop table if exists modelinstance cascade;
create table modelinstance (
	instanceid		text primary key,
	modelid			uuid REFERENCES model(modelid) ON DELETE CASCADE,
	simulationtime		numeric,
	lastupdated		date,
	FOREIGN KEY (modelid) REFERENCES model(modelid)
);

-- Table to store FMU model instancevalues
drop table if exists modelinstancevalues cascade;
create table modelinstancevalues (
	modelid			uuid REFERENCES model(modelid) ON DELETE CASCADE,
	instanceid		text REFERENCES modelinstance(instanceid) ON DELETE CASCADE,
	varname			text,
	value			numeric,
	PRIMARY KEY (modelid, instanceid, varname),
	FOREIGN KEY (modelid) REFERENCES model(modelid),
	FOREIGN KEY (instanceid) REFERENCES modelinstance(instanceid)
);
-- TODO:
-- impelment variant type for value columm


-- FMU_CREATE FUNCITON
DROP FUNCTION IF EXISTS fmu_create (modelref, instanceid);
CREATE OR REPLACE FUNCTION fmu_create (modelref text, instanceid text default '')
RETURNS text AS
$$
from pyfmi import load_fmu
from pymodelica import compile_fmu
import fnmatch
import numpy as np
import pandas as pd
import shutil
import os
import re
import random
import uuid
import sys
from datetime import datetime

global instanceid

now = datetime.now()
current_time = now.strftime("%H_%M_%S")
dirName = 'pgFMU'+'_'+current_time
os.mkdir (dirName)

dirPath = os.path.dirname(os.path.abspath(dirName))

if not os.path.exists(modelref):
#-------parsing modelica file
	filename = dirPath + dirName +str(random.randint(0,1000000))+".mo"
	f= open(filename,"w+")
	f.write (modelref)
	f.close()
	package_name = None
	models = []
	#----- ignore comments (lines begining with //)
	pattern = re.compile("^\s*//.*")
	for line in modelref.splitlines():
		if pattern.match(line):  
		#-----ignore if comment
			continue
		if "package" in line:
			package_name = line.split()[1]
		if "model" in line:
			models.append(line.split()[1])
	model_name = package_name +'.'+ models[0]
	model1 = compile_fmu(model_name, filename, version = "2.0")
	model = load_fmu(model1)
else:
	extension = os.path.splitext(modelref)[1]
	if extension == '.fmu':
		model1 = modelref
		model = load_fmu(modelref)
		model_name = model.get_name()
	else:
		if extension == '.mo' or extension == '.mop':
			package_name = None
			models = []
			pattern = re.compile("^\s*//.*")
			with open(modelref, 'r') as fil:
				for line in fil:
					if pattern.match(line):  
						continue
					if "package" in line:
						package_name = line.split()[1]
					if "model" in line:
						models.append(line.split()[1])
			
			model_name = package_name +'.'+ models[0]
			model1 = compile_fmu(model_name, modelref, version = "2.0")
			model = load_fmu(model1)
		else:
			plpy.notice("Oops! Sometwhing is wrong with the arguments you provided. Please check again.")
head, tail = os.path.split(model1)
modeluuid = uuid.uuid4()
dst_path = dirPath+'/'+dirName+'/'+str(modeluuid)+'.fmu'
shutil.copyfile(model1, dst_path)

if instanceid == '':
	instanceid= model_name

#-----for Model table
description = model.get_description()
defaultendtime = model. get_default_experiment_stop_time()
defaultstarttime = model. get_default_experiment_start_time()
#-----for ModelInstance table
lastupdated = model.get_generation_date_and_time()
#-----------------------------names
vars_in_name = model.get_model_variables(causality = 2).keys()
vars_out_name = model.get_model_variables(causality = 3).keys()
#vars_state_name = model.get_states_list().keys()
all_params = model.get_model_variables(variability = 1, type = 0, filter = '[!_]*').keys()
params_with_dot = fnmatch.filter(all_params, '*.*')
vars_param_name = list(set(all_params) - set(params_with_dot))
merged_list_names = vars_in_name + vars_out_name + vars_param_name
#----------------------------types
a_name = ["input" for x in vars_in_name]
b_name = ["output" for x in vars_out_name]
d_name = ["parameter" for x in vars_param_name]
merged_list_types = a_name + b_name + d_name
#-------------------------values
vars_in_values = [x[0] for x in model.get(vars_in_name)]
vars_out_values = [x[0] for x in model.get(vars_out_name)]
param_values = [x[0] for x in model.get(vars_param_name)]
merged_list_values = vars_in_values + vars_out_values + param_values
#-------------------------values min
values_min = []
for i in range(len(merged_list_names)):
    min1 = model.get_variable_min(merged_list_names[i])
    if min1 > -1.79769313486e+307:
	values_min.append(min1)
    else:
	values_min.append(None)
#-------------------------values max
values_max = []
for i in range(len(merged_list_names)):
    max1 = model.get_variable_max(merged_list_names[i])
    if max1 < 1.79769313486e+307:
	values_max.append(max1)
    else:
	values_max.append(None)
#----------------------updating Model
plan = plpy.prepare("INSERT INTO model(modelid, name, description, defaultstarttime, defaultendtime, modelref) VALUES ($1, $2, $3, $4, $5, $6)", ["uuid", "text", "text", "float", "float", "text"])
plpy.execute(plan, [modeluuid, model_name, description, defaultstarttime, defaultendtime, dst_path])
#----------------------updating ModelInstance
plan = plpy.prepare("INSERT INTO modelinstance(instanceid, modelid, simulationtime, lastupdated) VALUES ($1, $2, $3, $4)", ["text", "uuid", "float", "date"])
plpy.execute(plan, [instanceid, modeluuid, defaultendtime, lastupdated])
#----------------------updating ModelVariable
for x, y, z, xx, yy in zip(merged_list_names, merged_list_types, merged_list_values, values_min, values_max):
    plan = plpy.prepare("INSERT INTO modelvariable(modelid, varname, vartype, initialvalue, minvalue, maxvalue) VALUES ($1, $2, $3::fmu_vartype, $4, $5, $6)", ["uuid", "text", "fmu_vartype", "float", "float", "float"])
    plpy.execute(plan, [modeluuid, x, y, z, xx, yy])
#----------------------updating ModelInstanceValues
for x, y, z, xx, yy in zip(merged_list_names, merged_list_types, merged_list_values, values_min, values_max):
    plan = plpy.prepare("INSERT INTO modelinstancevalues(modelid, instanceid, varname, value) VALUES ($1, $2, $3, $4)", ["uuid", "text", "text", "float"])
    plpy.execute(plan, [modeluuid, instanceid, x, z])

return instanceid
$$
LANGUAGE plpythonu;


-- FMU_COPY FUNCTION
DROP FUNCTION IF EXISTS fmu_copy (instanceid, instanceid2);
CREATE OR REPLACE FUNCTION fmu_copy (instanceid text, instanceid2 text default '')
RETURNS void AS
$$
BEGIN
IF instanceid2 = '' THEN
	instanceid2 = CONCAT($1,'-',random());
	
END IF;
INSERT INTO modelinstance (instanceid, modelid, simulationtime, lastupdated) SELECT $2, modelid, simulationtime, lastupdated FROM modelinstance WHERE modelinstance.instanceid = $1;
INSERT INTO modelinstancevalues (modelid, instanceid, varname, value) SELECT modelid, $2, varname, value FROM modelinstancevalues WHERE modelinstancevalues.instanceid = $1;
END;
$$
LANGUAGE plpgsql;

-- FMU_VARIABLES FUNCTION
DROP FUNCTION IF EXISTS fmu_variables (instanceid);
CREATE OR REPLACE FUNCTION fmu_variables (instanceid text)
RETURNS table(instanceid text, varname text, vartype fmu_vartype, initialvalue numeric, minvalue numeric, maxvalue numeric) AS
$$
SELECT modelinstance.instanceid, modelvariable.varname, modelvariable.vartype, modelvariable.initialvalue, modelvariable.minvalue, modelvariable.maxvalue FROM modelvariable, modelinstance WHERE modelinstance.instanceid = $1 AND modelinstance.modelid = modelvariable.modelid;
$$
LANGUAGE sql;

-- FMU_GET FUNCTION
DROP FUNCTION IF EXISTS fmu_get (instanceid, varname);
CREATE OR REPLACE FUNCTION fmu_get (instanceid text, varname text)
RETURNS TABLE(initialvalue numeric, minvalue numeric, maxvalue numeric) AS
$$
SELECT modelvariable.initialvalue, modelvariable.minvalue, modelvariable.maxvalue FROM modelvariable, modelinstance WHERE modelinstance.instanceid=$1 AND modelinstance.modelid = modelvariable.modelid AND modelvariable.varname = $2;
$$
LANGUAGE sql;

-- FMU_SET_INITIAL FUNCTION
DROP FUNCTION IF EXISTS fmu_set_initial (instanceid, varname, initialvalue);
CREATE OR REPLACE FUNCTION fmu_set_initial (instanceid text, varname text, initialvalue numeric)
RETURNS text AS
$$
BEGIN
PERFORM modelid FROM modelinstance WHERE modelinstance.instanceid = $1;
UPDATE modelvariable SET initialvalue = $3 FROM (SELECT * FROM modelinstance) AS f WHERE f.instanceid=$1 AND modelvariable.varname = $2;
UPDATE modelinstancevalues SET value = $3 WHERE modelinstancevalues.modelid = modelid AND modelinstancevalues.instanceid = $1 AND modelinstancevalues.varname = $2;
RETURN instanceid;
END;
$$
LANGUAGE plpgsql;

-- FMU_SET_MINIMUM FUNCTION
DROP FUNCTION IF EXISTS fmu_set_minimum (instanceid, varname, minvalue);
CREATE OR REPLACE FUNCTION fmu_set_minimum (instanceid text, varname text, minvalue numeric)
RETURNS text AS
$$
UPDATE modelvariable SET minvalue = $3 FROM (SELECT * FROM modelinstance) AS f WHERE f.instanceid=$1 AND modelvariable.varname = $2 RETURNING f.instanceid;
$$
LANGUAGE sql;

-- FMU_SET_MAXIMUM FUNCTION
DROP FUNCTION IF EXISTS fmu_set_maximum (instanceid, varname, maxvalue);
CREATE OR REPLACE FUNCTION fmu_set_maximum(instanceid text, varname text, maxvalue numeric)
RETURNS text AS
$$
UPDATE modelvariable SET maxvalue = $3 FROM (SELECT * FROM modelinstance) AS f WHERE f.instanceid=$1 AND modelvariable.varname = $2 RETURNING f.instanceid;
$$
LANGUAGE sql;

-- FMU_DELETE_INSTANCE FUNCTION
DROP FUNCTION IF EXISTS fmu_delete_instance (instanceid);
CREATE OR REPLACE FUNCTION fmu_delete_instance (instanceid text)
RETURNS text AS
$$
DECLARE
    result text := 'Model instance deleted!';
BEGIN
IF EXISTS (SELECT 1 FROM modelinstance WHERE modelinstance.instanceid = $1 LIMIT 1) THEN
	DELETE FROM modelinstance WHERE modelinstance.instanceid = $1;
	RETURN result;
ELSE
	result := 'There is no model instance with such ID!';
	RETURN result;
END IF;
END;
$$
LANGUAGE plpgsql;

-- FMU_DELETE_MODEL FUNCITON
DROP FUNCTION IF EXISTS fmu_delete_model (instanceid);
CREATE OR REPLACE FUNCTION fmu_delete_model (instanceid text)
RETURNS text AS
$$
import fnmatch
import shutil
import os
import re
import uuid
import sys

#--------------------retrieve uuid
plan = plpy.prepare ("SELECT modelid FROM modelinstance WHERE instanceid = $1 ", ["text"])
rv = plpy.execute(plan, [instanceid])
if len(rv)==0:
	plpy.notice("There is no model with such ID!")
else:
	uuid1 = rv[0].values()[0]
	uuidreal = uuid.UUID(uuid1)
#--------------------retrieve path of the model with that uuid
plan = plpy.prepare ("SELECT modelref FROM model WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])
modelref = rv[0].values()[0]
#--------------------delete records from the tables
plan = plpy.prepare ("DELETE FROM model WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])
#--------------------delete files
head, tail = os.path.split(modelref)
shutil.rmtree(head)

return "Model successfully deleted!"
$$
LANGUAGE plpythonu;

-- FMU_SIMULATE FUNCTION
DROP FUNCTION IF EXISTS fmu_simulate (instanceid, query, timefrom, timeto);
CREATE OR REPLACE FUNCTION fmu_simulate (instanceid text, query text default '', timefrom numeric default null, timeto numeric default null)
RETURNS table (simulationtime numeric, instanceid text, varname text, varvalue numeric) AS
$$
import numpy as N
import pandas as pd
from pyfmi import load_fmu
import uuid
import os

global timefrom
global timeto
global query

#--------------------retrieve uuid + load fmu with that uuid
plan = plpy.prepare ("SELECT modelid FROM modelinstance WHERE instanceid = $1 ", ["text"])
rv = plpy.execute(plan, [instanceid])
uuid1 = rv[0].values()[0]
uuidreal = uuid.UUID(uuid1)
#--------------------load fmu with that uuid, retrieve defaultstarttime and defaultendtime
plan = plpy.prepare ("SELECT modelref, defaultstarttime, defaultendtime FROM model WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])
fmu_path = rv[0].values()[1]
defaultstarttime = rv[0].values()[0]
defaultendtime = rv[0].values()[2]
head, tail = os.path.split(fmu_path)
model = load_fmu(fmu_path)
#--------------------retrieve vars with this uuid
plan = plpy.prepare ("SELECT varname, vartype, initialvalue FROM modelvariable WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])

varname = [x["varname"] for x in rv]
vartype = [x["vartype"] for x in rv]
v_values = [x["initialvalue"] for x in rv]
df1 = pd.DataFrame({'varname': varname, 'vartype': vartype, 'values': v_values})

plan = plpy.prepare ("SELECT varname, value FROM modelinstancevalues WHERE modelid = $1 AND instanceid = $2 ", ["uuid", "text"])
rv = plpy.execute(plan, [uuidreal, instanceid])
varnames1 = [x["varname"] for x in rv]
values1 = [x["value"] for x in rv]
df2 = pd.DataFrame({'varname': varnames1, 'values': values1})

#--------------------update for params only
pars = df1.loc[df1['vartype'] == 'parameter']
pars.set_index('varname', inplace = True)
df2.set_index('varname', inplace = True)
pars.update(df2)
df1.set_index('varname', inplace = True)
df1.update(pars)
pars.reset_index(inplace = True)
df2.reset_index(inplace = True)
df1.reset_index(inplace = True)
#--------------------update model with new params values
k = len(df1['varname'])
for x in range(k):
	model.set(str(df1['varname'][x]), df1['values'][x])
#--------------------retrieving input vars list
ins = df1.loc[df1['vartype'] == 'input']
inputs = ins['varname'].tolist()
#--------------------retrieving output vars list
outs = df1.loc[df1['vartype'] == 'output']
outputs = outs['varname'].tolist()
#----------------------------- fetching measurements data
if query != '':
	rv = plpy.execute (query)
	headers = rv.colnames()
	measurements = []
	for i in range(rv.nrows()):
		measurements.append(rv[i])
	df = pd.DataFrame.from_dict(measurements)
	headers_df = df.columns.values.tolist()
	df = df.astype(float)
else:
	df = pd.DataFrame()
	headers = inputs + outputs
#----------------------------- dealing with timefrom and timeto
if timefrom == None:
	timefrom = defaultstarttime
if timeto == None:
	if not df.empty:
		timeto = len(df)
	else:
		timeto = defaultendtime
#----------------------------- creating input object
if query != '':
	t = N.linspace(int(timefrom), int(timeto)-1, int(timeto-timefrom))
	m = []
	for i in range (len(inputs)):
		if inputs[i] in headers:
			s = headers.index(inputs[i])
			m.append(df[df.columns[s]])
	m = N.asarray(m)
	u_traj = N.transpose(N.vstack((t, m[0:, int(timefrom):int(timeto)])))
	input_object = (inputs, u_traj)
	res = model.simulate(start_time = timefrom, final_time = timeto, input=input_object)
else:
	res = model.simulate(start_time = timefrom, final_time = timeto)
	t = res['time']
k=[]
k_names = []
for i in range (len(outputs)):
    if outputs[i] in headers:
        q = headers.index(outputs[i])
        k_names.append(outputs[i])

y = pd.DataFrame()
for i in range (len(k_names)):
	y[k_names[i]] = res[k_names[i]]
simulations = y.reindex(t)

for m in range(len(k_names)):
	plan = plpy.prepare ("UPDATE modelinstancevalues SET value = $4 WHERE modelid=$1 AND instanceid = $2 AND varname = $3;", ["uuid", "text", "text", "float"])
	rv = plpy.execute(plan, [uuidreal, instanceid , k_names[m], res[k_names[m]][len(simulations)-1]])

#--------------------------------- returning result
result = []
for i in range (len(simulations)):
	for m in range(len(k_names)):
		result.append((t[i], instanceid, k_names[m], res[k_names[m]][i]))
return result
$$
LANGUAGE plpythonu;


--FMU_PAREST FUNCTION
DROP FUNCTION IF EXISTS fmu_parest (instanceids, querys, variable, pars, threshold);
CREATE OR REPLACE FUNCTION fmu_parest (instanceids text[], querys text[], variable text, pars text[] default '{}', threshold numeric default 0)
RETURNS SETOF numeric AS
$$
import numpy as N
import pandas as pd
from pyfmi import load_fmu
from modestpy import Estimation
import uuid
import os
from numpy import linalg as LA

global pars
global variable
#--------------------calculate L2 norm (subfunc)
def norm_calc(ideal):
	s = LA.norm(ideal)
	plpy.notice("Calculating L2 norm for model")
	plpy.notice(s)
	plpy.notice("----------------------------")
	return s
#--------------------- retrieving measurements dataframe
def retrieve_ideal(query):
	rv = plpy.execute (querys[0])
	headers = rv.colnames()
	measurements = []
	for i in range(rv.nrows()):
		measurements.append(rv[i])
	ideal = pd.DataFrame.from_dict(measurements)
	return ideal
#--------------------- calculating parameters (global search)
def global_search():
	session = Estimation(workdir, fmu_path, inp, known, est, ideal_final, lp_len = int(lp_len_trial),
		lp_n = int((trn_t1-trn_t0)/lp_len_trial),
		lp_frame = (int(trn_t0), int(trn_t1)),
		vp = (int(vld_t0), int(vld_t1)),
		methods = ('GA', 'SCIPY'),
		ga_opts = ga_opts,
		scipy_opts = scipy_opts,
		ftype = 'RMSE')
	estimates = session.estimate()
	err, res = session.validate()
	plpy.notice("Printing estimated param values for model instance")
	plpy.notice(estimates)
	plpy.notice("----------------------------")
	plpy.notice("Printing error values for model instance")
	plpy.notice(err)
	plpy.notice("----------------------------")
	return err, estimates
	#--------------------- calculating parameters (local search)
def local_search():
	session = Estimation(workdir, fmu_path, inp, known, est, ideal_final, lp_len = int(lp_len_trial),
		lp_n = int((trn_t1-trn_t0)/lp_len_trial),
		lp_frame = (int(trn_t0), int(trn_t1)),
		vp = (int(vld_t0), int(vld_t1)),
		methods = ('SCIPY', ),
		scipy_opts = scipy_opts,
		ftype = 'RMSE')
	estimates = session.estimate()
	err, res = session.validate()
	plpy.notice("Printing estimated param values for model instance")
	plpy.notice(estimates)
	plpy.notice("----------------------------")
	plpy.notice("Printing error values for model instance")
	plpy.notice(err)
	plpy.notice("----------------------------")
	return err, estimates

#--------------------retrieve uuid of ID[0] + load fmu with that uuid
plan = plpy.prepare ("SELECT modelid FROM modelinstance WHERE instanceid = $1 ", ["text"])
rv = plpy.execute(plan, [instanceids[0]])
uuid0 = rv[0].values()[0]
uuidreal = uuid.UUID(uuid0)
#--------------------point to the fmu with that uuid
plan = plpy.prepare ("SELECT modelref FROM model WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])
fmu_path = rv[0].values()[0]
head, tail = os.path.split(fmu_path)
workdir = head
#--------------------retrieve vars with this uuid
plan = plpy.prepare ("SELECT varname, vartype, initialvalue, minvalue, maxvalue FROM modelvariable WHERE modelid = $1 ", ["uuid"])
rv = plpy.execute(plan, [uuidreal])
varname = []
vartype = []
initialvalue = []
minvalue1 = []
maxvalue1 = []
minvalue = []
maxvalue = []
for x in rv:
    varname.append(x["varname"])
    vartype.append(x["vartype"])
    initialvalue.append(x["initialvalue"])
    minvalue1.append(x["minvalue"])
    maxvalue1.append(x["maxvalue"])
initialvalue = [float(i) for i in initialvalue]
for i in range(len(minvalue1)):
	if minvalue1[i] != None:
		minvalue.append(float(minvalue1[i]))
	else:
		minvalue.append(-float('Inf'))
for i in range(len(maxvalue1)):
	if maxvalue1[i] != None:
		maxvalue.append(float(maxvalue1[i]))
	else:
		maxvalue.append(float('Inf'))
				
p = zip(varname, vartype, initialvalue, minvalue, maxvalue)
vars = pd.DataFrame({'varname': varname, 'vartype': vartype, 'initialvalue': initialvalue, 'minvalue': minvalue, 'maxvalue': maxvalue})
#--------------------- input variable
inps = vars['vartype'] == "input"
params = vars['vartype'] == "parameter"
#--------------------- k = all the params from dataframe
k = vars[inps]
l = vars[params]
input_var = k['varname'].tolist()
params_names = l['varname'].tolist()
#--------------------- if param names are not given - fetching all the params
if pars == '{}':
	pars = params_names
ideal = retrieve_ideal(querys[0])
#--------------------- preparing for params estimation
headers_df = ideal.columns.values.tolist()
ideal_final = ideal[['time', variable]]
ideal_time = ideal[['time']]
#--------------------- preparing for params estimation (creating input object)
m = []
m.append(ideal['time'])
for i in range (len(input_var)):
    if input_var[i] in headers_df:
        b = headers_df.index(input_var[i])
        m.append(ideal[ideal.columns[b]])

inp_T = pd.DataFrame(m)
inp = inp_T.transpose()
inp.set_index('time', inplace = True)
inp = inp.astype(float)

ideal.set_index('time', inplace = True)
ideal = ideal.astype(float)
s0 = norm_calc (ideal)
ideal_final.set_index('time', inplace = True)
ideal_final = ideal_final.astype(float)

#--------------------- retrieving est params
h = dict(zip(l['varname'].tolist(), zip(l['initialvalue'].tolist(), l['minvalue'].tolist(), l['maxvalue'].tolist())))
f = dict(zip(l['varname'].tolist(), l['initialvalue'].tolist()))
est = {}
for i in range(len(pars)):
    p = dict ([(pars[i], h.get(pars[i]))])
    est.update(p)

#--------------------- retrieving known params
remaining = list(set(params_names) - set(pars))
known = {}
for i in range(len(remaining)):
    m = dict ([(remaining[i], f.get(remaining[i]))])
    known.update(m)

ga_opts = {'maxiter': 50, 'pop_size': 20, 'tol': 1e-6, 'mut': 0.10, 'mut_inc': 0.33, 'uniformity': 0.5, 'look_back':50, 'lhs': False}
scipy_opts = {
    'solver': 'SLSQP',
    'options': {'maxiter': 50, 'tol': 1e-12, 'eps': 1e-3}
}
#--------------------- training and validation period
trn_t0 = 0
trn_t1 = trn_t0 + 0.8 * ideal_time['time'].iloc[-1]
vld_t0 = trn_t0
vld_t1 = trn_t0 + ideal_time['time'].iloc[-1]
lp_len_trial = (ideal_time['time'].iloc[-1])/10

error0, estimates0 = global_search()

for key, value in estimates0.items():
	plan = plpy.prepare ("UPDATE modelinstancevalues SET value = $4 WHERE modelid=$1 AND instanceid = $2 AND varname = $3;", ["uuid", "text", "text", "float"])
	rv = plpy.execute(plan, [uuidreal, instanceids[0], key, value])

#------------------------------------------------------------------ estimating the remaining models
for i in range(1, len(instanceids)):
	#--------------------- retrieving dataframe
	ideal = retrieve_ideal(querys[i])
	#--------------------- preparing for params estimation
	headers_df = ideal.columns.values.tolist()
	ideal_final = ideal[['time', variable]]

	m = []
	m.append(ideal['time'])
	for i in range (len(input_var)):
		if input_var[i] in headers_df:
			b = headers_df.index(input_var[i])
			m.append(ideal[ideal.columns[b]])

	inp_T = pd.DataFrame(m)
	inp = inp_T.transpose()
	ideal.set_index('time', inplace = True)
	ideal = ideal.astype(float)
	ideal_final.set_index('time', inplace = True)
	ideal_final = ideal_final.astype(float)
	inp.set_index('time', inplace = True)
	inp = inp.astype(float)	

	plan = plpy.prepare ("SELECT modelid FROM modelinstance WHERE instanceid = $1 ", ["text"])
	rv = plpy.execute(plan, [instanceids[i]])
	uuid2 = rv[0].values()[0]
	uuidreal2 = uuid.UUID(uuid2)

	if uuid2 != uuid0:
		plpy.notice("Instances are of different types (uuids are not the same), cannot use multi-instance optimization")
		error, estimates = global_search()
	else:
		plpy.notice("Instances are of the same uuid, calculating the similarity between time series")
		s1 = norm_calc(ideal)
		delta = abs(1-s1/s0)
		plpy.notice("The difference in time series is")
		plpy.notice(delta)
		plpy.notice("----------------------------")
		if delta > threshold:
			plpy.notice("The measurements are not similar, difference is greater than the threshold")
			error, estimates = global_search()
		else:
			plpy.notice("Updating initial param values of i-th instance with optimal param values of instance0")
			for key, value in est.items():
				value1 = (estimates0.get(key),) + value[1:]
				est[key] = value1
			error, estimates = local_search()

result = error0.values()[1::2] + error.values()[1::2]
return result    
$$
LANGUAGE plpythonu;
