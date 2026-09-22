@echo off
set "JAVA_HOME=C:\Users\carpe\.jdks\jdk-17.0.20.1+1"
set "ADB_PATH=C:\Users\carpe\AppData\Local\Android\Sdk\platform-tools\adb.exe"
set "PATH=%JAVA_HOME%\bin;C:\Users\carpe\AppData\Local\Android\Sdk\platform-tools;C:\Windows\system32;C:\Windows"
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set ALL_PROXY=
set all_proxy=
cd /d "c:\Users\carpe\SIH\android"
echo [1/3] Building Debug APK with Gradle...
call gradlew.bat assembleDebug --offline --console=plain
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gradle build failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo [2/3] Build succeeded! Checking for connected ADB devices...
if exist "%ADB_PATH%" (
    "%ADB_PATH%" devices | findstr /R "device$" >nul
    if %ERRORLEVEL% EQU 0 (
        echo [3/3] Found connected device! Installing app-debug.apk...
        "%ADB_PATH%" install -r app\build\outputs\apk\debug\app-debug.apk
        if %ERRORLEVEL% EQU 0 (
            echo [SUCCESS] APK installed successfully! Launching IDR app...
            "%ADB_PATH%" shell am start -n com.recursiveminds.idr/.ui.MainActivity
        ) else (
            echo [WARNING] ADB install failed. Please check device screen for install prompt.
        )
    ) else (
        echo [INFO] No device attached in 'device' mode. APK is ready at:
        echo app\build\outputs\apk\debug\app-debug.apk
    )
) else (
    echo [INFO] ADB not found at %ADB_PATH%. APK is ready at:
    echo app\build\outputs\apk\debug\app-debug.apk
)
