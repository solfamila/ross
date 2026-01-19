@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\forfr\Downloads\trade
if exist cpp_realtime_ocr\build rmdir /s /q cpp_realtime_ocr\build
cmake -S cpp_realtime_ocr -B cpp_realtime_ocr\build -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:\vcpkg\installed\x64-windows\share\opencv4 -DProtobuf_DIR=C:\vcpkg\installed\x64-windows\share\protobuf
endlocal
