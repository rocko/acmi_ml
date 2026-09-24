@echo off
setlocal

REM ============================================================
REM ACMI ML - Build all datasets
REM ============================================================
REM Expected input files:
REM
REM raw\a10_landing.acmi
REM raw\hornet_speed_change.acmi
REM raw\hornet_loop.acmi
REM raw\hornet_long_turn.acmi
REM raw\hornet_erratic.acmi
REM
REM Hornet aircraft ID: 1
REM A-10 aircraft ID:   16777472
REM ============================================================

cd /d "%~dp0"

REM ------------------------------------------------------------
REM Prefer repository virtual environment if available.
REM ------------------------------------------------------------

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo.
echo ============================================================
echo ACMI ML DATASET BUILDER
echo ============================================================
echo Python: %PYTHON%
echo.

REM ------------------------------------------------------------
REM Create output directory.
REM ------------------------------------------------------------

if not exist "datasets" (
    mkdir "datasets"
)

REM ------------------------------------------------------------
REM Check required input files.
REM ------------------------------------------------------------

call :check_file "raw\sortie_05.acmi"
if errorlevel 1 goto :error

call :check_file "raw\sortie_04.acmi"
if errorlevel 1 goto :error

call :check_file "raw\sortie_03.acmi"
if errorlevel 1 goto :error

call :check_file "raw\sortie_02.acmi"
if errorlevel 1 goto :error

call :check_file "raw\sortie_01.acmi"
if errorlevel 1 goto :error


REM ============================================================
REM SORTIE 01 - A-10 landing
REM ============================================================

echo.
echo [1/5] Building A-10 landing dataset...

"%PYTHON%" build_flight_dataset.py ^
    "raw\sortie_05.acmi" ^
    --aircraft-id 16777472 ^
    --start-time 1200 ^
    --end-time 1230 ^
    --window-seconds 5 ^
    --stride-seconds 1 ^
    --output "datasets\sortie_05.csv"

if errorlevel 1 goto :error


REM ============================================================
REM SORTIE 02 - Hornet speed change
REM ============================================================

echo.
echo [2/5] Building Hornet speed-change dataset...

"%PYTHON%" build_flight_dataset.py ^
    "raw\sortie_04.acmi" ^
    --aircraft-id 1 ^
    --window-seconds 5 ^
    --stride-seconds 1 ^
    --output "datasets\sortie_04.csv"

if errorlevel 1 goto :error


REM ============================================================
REM SORTIE 03 - Hornet loop
REM ============================================================

echo.
echo [3/5] Building Hornet loop dataset...

"%PYTHON%" build_flight_dataset.py ^
    "raw\sortie_03.acmi" ^
    --aircraft-id 1 ^
    --window-seconds 5 ^
    --stride-seconds 1 ^
    --output "datasets\sortie_03.csv"

if errorlevel 1 goto :error


REM ============================================================
REM SORTIE 04 - Hornet long turn
REM ============================================================

echo.
echo [4/5] Building Hornet long-turn dataset...

"%PYTHON%" build_flight_dataset.py ^
    "raw\sortie_02.acmi" ^
    --aircraft-id 1 ^
    --window-seconds 5 ^
    --stride-seconds 1 ^
    --output "datasets\sortie_02.csv"

if errorlevel 1 goto :error


REM ============================================================
REM SORTIE 05 - Hornet erratic flight
REM ============================================================

echo.
echo [5/5] Building Hornet erratic-flight dataset...

"%PYTHON%" build_flight_dataset.py ^
    "raw\sortie_01.acmi" ^
    --aircraft-id 1 ^
    --window-seconds 5 ^
    --stride-seconds 1 ^
    --output "datasets\sortie_01.csv"

if errorlevel 1 goto :error


REM ============================================================
REM Combine all sortie CSVs
REM ============================================================

echo.
echo Combining datasets...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$files = Get-ChildItem 'datasets\sortie_*.csv' | Sort-Object Name;" ^
    "$rows = foreach ($file in $files) { Import-Csv $file.FullName };" ^
    "$rows | Export-Csv 'datasets\all_sorties.csv' -NoTypeInformation -Encoding UTF8"

if errorlevel 1 goto :error


REM ============================================================
REM Show final label distribution
REM ============================================================

echo.
echo ============================================================
echo FINAL LABEL DISTRIBUTION
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Import-Csv 'datasets\all_sorties.csv' |" ^
    "Group-Object label |" ^
    "Sort-Object Count -Descending |" ^
    "Select-Object Count, Name |" ^
    "Format-Table -AutoSize"


echo.
echo ============================================================
echo DONE
echo ============================================================
echo.
echo Generated:
echo   datasets\sortie_01.csv
echo   datasets\sortie_02.csv
echo   datasets\sortie_03.csv
echo   datasets\sortie_04.csv
echo   datasets\sortie_05.csv
echo   datasets\all_sorties.csv
echo.

exit /b 0


REM ============================================================
REM Helpers
REM ============================================================

:check_file
if not exist "%~1" (
    echo ERROR: Missing input file:
    echo   %~1
    exit /b 1
)
exit /b 0


:error
echo.
echo ============================================================
echo ERROR
echo ============================================================
echo Dataset generation failed.
echo.
exit /b 1