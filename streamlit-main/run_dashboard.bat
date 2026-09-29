@echo off
cd /d "%~dp0"
echo Starting Streamlit dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false > streamlit.log 2> streamlit.err
echo Streamlit exited with code %ERRORLEVEL%.>> streamlit.err
