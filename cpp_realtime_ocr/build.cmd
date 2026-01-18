@echo off
setlocal

REM Builds with Ninja using VS DevCmd + CUDA environment.
REM Usage:
REM   build.cmd [Release|Debug]

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

REM Avoid cmd.exe PATH-length limits: shrink PATH before VsDevCmd extends it.
set "ORIGINAL_PATH=%PATH%"
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem"

REM The project CMakeLists uses an absolute nvcc path via CMAKE_CUDA_COMPILER.

REM Try PATH first (portable)
set "CMAKE_EXE=cmake"
set "NINJA_EXE=ninja"

where cmake >nul 2>nul
if errorlevel 1 set "CMAKE_EXE=C:\Users\forfr\.mcuxpressotools\cmake-3.30.0-windows-x86_64\bin\cmake.exe"

where ninja >nul 2>nul
if errorlevel 1 set "NINJA_EXE=C:\Users\forfr\.mcuxpressotools\ninja-1.12.1\ninja.exe"

REM Locate VsDevCmd.bat (Insiders preferred, then VS2022 editions)
set "VSDEVCMD="

if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat" (
	set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
)

if "%VSDEVCMD%"=="" (
	for %%E in (Enterprise Professional Community BuildTools) do (
		if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\Common7\Tools\VsDevCmd.bat" (
			set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\%%E\Common7\Tools\VsDevCmd.bat"
		)
	)
)

if "%VSDEVCMD%"=="" (
	echo ERROR: Could not find VsDevCmd.bat.
	echo Install Visual Studio with "Desktop development with C++" + Windows 11 SDK.
	exit /b 1
)

call "%VSDEVCMD%" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b %errorlevel%

cd /d "%~dp0build"
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" .. -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 exit /b %errorlevel%

"%NINJA_EXE%"
exit /b %errorlevel%
