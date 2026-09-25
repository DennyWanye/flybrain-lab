param(
  [string]$WslProject = "/home/denny/projects/flybrain_lab_4spark_v0_1",
  [int]$Port = 8765
)
$command = "cd '$WslProject' && PYTHONPATH='$WslProject' python3 -m flyview serve --project-root '$WslProject' --host 127.0.0.1 --port $Port"
Start-Process wsl.exe -ArgumentList @('-e','bash','-lc',$command)
Start-Sleep -Seconds 1
Start-Process "http://127.0.0.1:$Port"
