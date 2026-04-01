@echo off
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

echo Installing nb_conda_kernels in base environment...
call conda install -n base nb_conda_kernels -y

echo Creating NoteEnv environment...
call conda env create -f Notepad/NoteEnv.yml

echo.
echo Setup complete. To start:
echo   jupyter lab
pause
