$LOG  = "d:\Users\Namhyeonseung\Desktop\파일\대학 수업\3학년\1학기\딥러닝 기초\프로젝트\results\train_log_model_v13.txt"
$LAST = "d:\Users\Namhyeonseung\Desktop\파일\대학 수업\3학년\1학기\딥러닝 기초\프로젝트\results\monitor_v13_last.txt"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$lines = [System.IO.File]::ReadAllLines($LOG, [System.Text.Encoding]::UTF8)
$last  = if (Test-Path $LAST) { ([System.IO.File]::ReadAllText($LAST, [System.Text.Encoding]::UTF8)).Trim() } else { "" }

# 완료된 에폭 파악
$done_epochs = @()
foreach ($l in $lines) {
    if ($l -match 'Epoch (\d+)/20 \|.*Val') { $done_epochs += [int]$Matches[1] }
}
$last_done       = if ($done_epochs.Count -gt 0) { $done_epochs[-1] } else { 0 }
$last_done_shown = if ($last -match '^(\d+),done$') { [int]$Matches[1] } else { 0 }

# 새 완료 에폭 출력
for ($ep = $last_done_shown + 1; $ep -le $last_done; $ep++) {
    $summary = $lines | Where-Object { $_ -match "Epoch ${ep}/20 \|.*Val" } | Select-Object -First 1
    if ($summary) { Write-Host $summary }
    $last = "${ep},done"
    [System.IO.File]::WriteAllText($LAST, $last, [System.Text.Encoding]::UTF8)
}

if ($last -eq "20,done") {
    Write-Host "[COMPLETE] Epoch 20/20 complete!"
    exit 100
}

# 현재 epoch/step 파악 (tqdm 라인)
$cur_ep = 0; $cur_step = 0
foreach ($l in $lines) {
    if ($l -match 'Epoch (\d+)/20 \[Train v9\].*?\s(\d+)/1859') {
        $cur_ep = [int]$Matches[1]
        $cur_step = [int]$Matches[2]
    }
}

if ($cur_ep -eq 0) { Write-Host "[INFO] no active epoch"; exit 0 }

# 현재 step -> 구간 -> milestone label
$pct = $cur_step / 1859.0
$ms  = if ($pct -ge 0.75) { "75%" } elseif ($pct -ge 0.50) { "50%" } elseif ($pct -ge 0.25) { "25%" } else { "0%" }
$key = "${cur_ep},${ms}"

# 이미 보여준 milestone이거나 에폭 완료면 스킵
if ($key -eq $last -or $last -eq "${cur_ep},done") { exit 0 }

# 해당 milestone 라인 찾아서 출력
$m_line = $lines | Where-Object { $_ -match "\[MILESTONE\] Epoch ${cur_ep}/20 ${ms}" } | Select-Object -Last 1
if ($m_line) {
    Write-Host $m_line
} else {
    $pct_str = ($pct * 100).ToString("F1")
    Write-Host "[Epoch $cur_ep/20] $ms range (step=$cur_step/1859, $pct_str%) - collecting..."
}
[System.IO.File]::WriteAllText($LAST, $key, [System.Text.Encoding]::UTF8)
exit 0
