@echo off
REM Script biên dịch mạng lưới SUMO dành cho Windows
REM Chạy bằng cách double-click vào file này hoặc gõ trên terminal: .\build_network.bat

echo [INFO] Building SUMO network...

netconvert --node-files=nodes.nod.xml ^
           --edge-files=edges.edg.xml ^
           --output-file=network.net.xml

REM Kiểm tra mã lỗi trả về từ lệnh netconvert
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Network built successfully: network.net.xml
) else (
    echo [ERROR] Build failed! Check the XML files for typos or logic errors.
    exit /b %ERRORLEVEL%
)

REM Dừng màn hình để đọc log nếu chạy bằng cách double-click trực tiếp
pause