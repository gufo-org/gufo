@echo off
rem Loads the MSVC x64 environment (with Visual Studio's CMake and Ninja) for
rem gufo Windows builds, then runs the given command. vswhere finds any
rem edition and version that has the C++ x64 tools.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere.exe not found; install Visual Studio with the C++ x64 tools & exit /b 1)
set "VSROOT="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo Visual Studio with the C++ x64 tools was not found & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
%*
