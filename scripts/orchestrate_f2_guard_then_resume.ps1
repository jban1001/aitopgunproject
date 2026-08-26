param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [Parameter(Mandatory = $true)]
    [string]$InitialBundle,
    [string]$Python = $(if ($env:AIP_PYTHON) { $env:AIP_PYTHON } else { "python" }),
    [int]$MinIterations = 80,
    [int]$PlateauPatience = 25,
    [double]$ScoreThreshold = 0.82
)

$ErrorActionPreference = "Stop"
$LogRoot = Join-Path $Root "artifacts\logs\team01"
$OrchestratorLog = Join-Path $LogRoot "F2_guard_orchestrator.log"
$Experiment = Join-Path $Root "experiments\team01_F2_guard.yaml"

function Write-Status([string]$Message) {
    $line = "[{0}] [F2-guard] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $OrchestratorLog -Value $line -Encoding utf8
}

function Get-Score([string]$Tag) {
    $csv = Join-Path $LogRoot "$Tag\training_log.csv"
    $json = & $Python (Join-Path $Root "scripts\f_transfer_score.py") $csv --threshold $ScoreThreshold --min-iteration $MinIterations
    return $json | ConvertFrom-Json
}

function Get-PeakBundle([string]$Tag, [int]$PeakIteration) {
    $modelRoot = Join-Path $Root "artifacts\models\team01\$Tag"
    $bundles = @(Get-ChildItem -LiteralPath $modelRoot -Directory -Filter "bundle_*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $iteration = [int]($_.Name.Substring("bundle_".Length))
            [pscustomobject]@{ Path = $_.FullName; Iteration = $iteration }
        } | Sort-Object Iteration)
    if ($bundles.Count -eq 0) { return $null }
    $eligible = @($bundles | Where-Object { $_.Iteration -le $PeakIteration })
    if ($eligible.Count -gt 0) { return $eligible[-1].Path }
    return $bundles[0].Path
}

function Get-LastCheckpoint([string]$Tag) {
    $checkpointRoot = Join-Path $Root "artifacts\checkpoints\team01\$Tag"
    $checkpoints = @(Get-ChildItem -LiteralPath $checkpointRoot -Directory -Filter "checkpoint_*" -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($checkpoints.Count -eq 0) { return $null }
    return $checkpoints[-1].FullName
}

function Invoke-GuardRun(
    [string]$Tag,
    [string]$Parent,
    [bool]$RestoreFullState,
    [int]$IterationCap
) {
    $stdout = Join-Path $LogRoot "run_$Tag.log"
    $stderr = Join-Path $LogRoot "run_$Tag.err.log"
    $stopFile = Join-Path $LogRoot "$Tag\plateau_stop.request"
    Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $Parent)) { throw "Parent state missing: $Parent" }

    $arguments = @(
        "train_rllib.py", "--algorithm", "sac", "--iterations", "$IterationCap",
        "--output-name", "team01", "--output-tag", $Tag,
        "--framework", "torch", "--lr", "3e-05", "--initial-alpha", "1e-05", "--gamma", "0.99",
        "--train-batch-size", "256", "--minibatch-size", "256", "--tau", "0.005",
        "--target-entropy", "-1.0", "--replay-buffer-capacity", "200000",
        "--model-fcnet-hiddens", "256,256", "--model-fcnet-activation", "relu",
        "--model-head-fcnet-hiddens=", "--model-head-fcnet-activation", "relu",
        "--observation-mode", "custom", "--observation-module", "student.team01_phase_observation",
        "--target-behavior-dll", "AIP_BASE_team_climb_dive_approach.dll", "--target-mode", "behavior_tree",
        "--max-engage-time", "200.0", "--episode-step-limit", "2000",
        "--num-env-runners", "2", "--num-envs-per-env-runner", "1",
        "--rollout-fragment-length", "auto", "--batch-mode", "truncate_episodes",
        "--lightweight-bundle-frequency", "10", "--native-checkpoint-frequency", "25", "--save-native-checkpoint",
        "--policy-probe-interval", "25", "--policy-probe-steps", "4", "--no-policy-probe-print",
        "--engagement-log-interval", "25", "--engagement-log-steps", "600", "--engagement-log-episodes", "6", "--no-engagement-log-print",
        "--stop-file", $stopFile, "--dashboard-logdir", "artifacts\dashboard",
        "--experiment-yaml", $Experiment
    )
    if ($RestoreFullState) { $arguments += @("--restore-checkpoint", $Parent) }
    else { $arguments += @("--init-bundle", $Parent) }

    $mode = if ($RestoreFullState) { "full-state continuation" } else { "peak-bundle transfer" }
    Write-Status "$Tag launch ($mode, cap=$IterationCap, min=$MinIterations) <- $Parent"
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

    $lastIteration = -1
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 120
        $process.Refresh()
        $score = Get-Score $Tag
        if ($score.current_iter -ne $null -and [int]$score.current_iter -ne $lastIteration) {
            $lastIteration = [int]$score.current_iter
            if ($score.ready) {
                Write-Status ("{0} iter={1} score={2:N3} transfer={3:N3} threat={4:N3} incoming={5:N3} survival={6:N3} offense={7:N3} plateau={8}" -f $Tag, $score.current_iter, $score.current.score, $score.current.transfer, $score.current.threat_clear, $score.current.incoming_clear, $score.current.survival, $score.current.offense, $score.plateau_age)
                if ([int]$score.current_iter -ge $MinIterations -and [bool]$score.quality_ok -and [int]$score.plateau_age -ge $PlateauPatience) {
                    Write-Status "$Tag qualified and plateaued; requesting graceful stop at iter=$($score.current_iter)"
                    New-Item -ItemType File -Force -Path $stopFile | Out-Null
                    break
                }
            }
        }
    }
    Wait-Process -Id $process.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 20
    $final = Get-Score $Tag
    $qualified = $false
    if ($final.ready) {
        $qualified = [bool]$final.quality_ok -and [int]$final.current_iter -ge $MinIterations
        $final | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $LogRoot "$Tag\F_transfer_stage_summary.json") -Encoding utf8
    }
    return [pscustomobject]@{ Tag = $Tag; Final = $final; Qualified = $qualified }
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
if (-not (Test-Path -LiteralPath $Experiment)) { throw "Experiment missing: $Experiment" }
Write-Status "start threshold=$ScoreThreshold min=$MinIterations plateau=$PlateauPatience parent=$InitialBundle"

$run = Invoke-GuardRun "F2_guard" $InitialBundle $false 180
if (-not $run.Qualified) {
    $checkpoint = Get-LastCheckpoint $run.Tag
    if (-not $checkpoint) { throw "F2_guard failed with no checkpoint" }
    Write-Status "F2_guard below gate; starting one full-state continuation"
    $run = Invoke-GuardRun "F2_guardb" $checkpoint $true 180
}

if (-not $run.Qualified) {
    $scoreText = if ($run.Final.ready) { "{0:N3}" -f $run.Final.current.score } else { "n/a" }
    Write-Status "$($run.Tag) failed after one continuation (score=$scoreText); chain halted"
    exit 0
}

$parent = Get-PeakBundle $run.Tag ([int]$run.Final.best.iter)
if (-not $parent) { throw "$($run.Tag) has no peak bundle" }
Write-Status "$($run.Tag) promoted from peak iter=$($run.Final.best.iter) score=$($run.Final.best.score) bundle=$parent"
Write-Status "resuming the original F ladder at F3"
& (Join-Path $Root "scripts\orchestrate_f_transfer_lane.ps1") -Root $Root -InitialBundle $parent -Python $Python -StartStage F3
