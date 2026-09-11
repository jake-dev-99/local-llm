 @echo on
 call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force >nul
 if errorlevel 1 exit /b %errorlevel%
 echo === compiler ===
 icx --version
 echo === sycl-ls ===
 sycl-ls.exe
 echo sycl-ls exit code: %errorlevel%
 echo === llama-server ===
 "C:\Users\AllenJ70\Development\local-llm\build\llama.cpp-win32-x64-sycl\bin\llama-server.exe" --list-devices
 echo llama-server exit code: %errorlevel%
 
./sycl-check.cmd
