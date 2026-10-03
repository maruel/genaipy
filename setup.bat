@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

cd "%~dp0"

if NOT EXIST venv python3 -m venv venv
if errorlevel 1 exit /b %errorlevel%
call venv\Scripts\activate.bat
if errorlevel 1 exit /b %errorlevel%

call python -m pip install -U pip
if errorlevel 1 exit /b %errorlevel%
call python -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%
