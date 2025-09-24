@echo off
echo ===================================
echo EXAM MONITORING SYSTEM
echo ===================================
echo.
echo Activating virtual environment...
call "exam_env\Scripts\activate.bat"
echo.
echo Starting exam monitor...
echo Press 'c' to calibrate exam zone, 'q' to quit
echo.
python exam_monitor_local.py
echo.
echo Exam monitoring session ended.
pause