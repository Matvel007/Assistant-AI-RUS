@echo off
powershell -Command "Get-ChildItem -Recurse -File | ForEach-Object { $_.Name } | Out-File -FilePath .\filelist.txt -Encoding utf8"
echo Готово.
pause