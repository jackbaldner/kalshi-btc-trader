@echo off
cd /d C:\Users\jackb\kalshi-btc-trader
C:\Users\jackb\AppData\Local\Python\bin\python.exe -m streamlit run dashboard.py --server.headless true >> logs\dashboard.log 2>&1
