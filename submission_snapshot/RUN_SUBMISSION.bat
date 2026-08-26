@echo off
setlocal
cd /d "%~dp0"

if defined AIP_PYTHON if exist "%AIP_PYTHON%" goto run_explicit
if exist "C:\Users\JUN\miniconda3\envs\aip\python.exe" goto run_local_aip
where python >nul 2>nul
if errorlevel 1 goto no_python

python student\submission.py
goto finished

:run_explicit
"%AIP_PYTHON%" student\submission.py
goto finished

:run_local_aip
"C:\Users\JUN\miniconda3\envs\aip\python.exe" student\submission.py
goto finished

:no_python
echo [ERROR] Python was not found.
echo Install requirements.txt, then run: python student\submission.py
pause
exit /b 1

:finished
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
