# pgFMU
An extension for traditional DBMS to work with model simulation, validation and calibration.

pgFMU is a solution for data analysts that need to work with both data and FMU-based simulation models in a single model- and data management environment. pgFMU facilitates user interaction with FMI-based model storage, model simulation, and parameter estimation tasks inside an SQL-based environment. For this purpose, pgFMU offers a number of convenient User Defined Functions (UDFs) accessible by simple SQL queries for each necessary operation.

# Installation

# Prerequisites:
1. A Linux-based distribution, with Kernel 3.19 or higher (test on Ubuntu 16.04), with superuser permissions.
2. Basic development tools: make, gcc, g++
3. PostgreSQL 9.6 or higher

# Third-party software

pgFMU relies on a number of third-party software, namely JModelica. please make sure to install JModelica firstly.

Python packages:
 - pyFMI
 - pymodelica
 - Modestpy

More information is coming.
