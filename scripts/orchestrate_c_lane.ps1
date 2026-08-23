param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [Parameter(Mandatory = $true)]
    [string]$InitialBundle,
    [string]$Python = $(if ($env:AIP_PYTHON) { $env:AIP_PYTHON } else { "python" }),
    [int]$MinIterations = 100,
    [int]$ContinuationMinIterations = 80,
    [int]$PlateauPatience = 25,
    [double]$ScoreThreshold = 0.72,
    [int]$AdoptC1Pid = 0,
    [ValidateSet("C1", "C2", "C3", "C4", "C5")]
    [string]$StartStage = "C1"
)

$ErrorActionPreference = "Stop"
$LogRoot = Join-Path $Root "artifacts\logs\team01"
$OrchestratorLog = Join-Path $LogRoot "C_series_orchestrator.log"
$AllStages = @("C1", "C2", "C3", "C4", "C5")
$startIndex = [array]::IndexOf($AllStages, $StartStage)
$Stages = @($AllStages[$startIndex..($AllStages.Count - 1)])

function Write-Status([string]$Message) {
    $line = "[{0}] [C] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $OrchestratorLog -Value $line -Encoding utf8
}

function Get-Score([string]$Tag, [int]$RunMinimum) {
    $csv = Join-Path $LogRoot "$Tag\training_log.csv"
    $json = & $Python (Join-Path $Root "scripts\c_curriculum_score.py") $csv --threshold $ScoreThreshold --min-iteration $RunMinimum
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

function Invoke-CTrainingRun(
    [string]$Tag,
    [string]$Experiment,
    [string]$Parent,
    [bool]$RestoreFullState,
    [int]$RunMinimum,
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
        # Match the proven D4 training topology. A local-only EnvRunner stalls
        # during SAC construction in this Windows/native-simulator setup.
        "--num-env-runners", "2", "--num-envs-per-env-runner", "1",
        "--rollout-fragment-length", "auto", "--batch-mode", "truncate_episodes",
        "--lightweight-bundle-frequency", "10", "--native-checkpoint-frequency", "25", "--save-native-checkpoint",
        "--policy-probe-interval", "25", "--policy-probe-steps", "4", "--no-policy-probe-print",
        "--engagement-log-interval", "25", "--engagement-log-steps", "600", "--engagement-log-episodes", "6", "--no-engagement-log-print",
        "--stop-file", $stopFile, "--dashboard-logdir", "artifacts\dashboard",
        "--experiment-yaml", $Experiment
    )
    if ($RestoreFullState) {
        $arguments += @("--restore-checkpoint", $Parent)
    } else {
        $arguments += @("--init-bundle", $Parent)
    }

    $mode = if ($RestoreFullState) { "full-state continuation" } else { "peak-bundle transfer" }
    Write-Status "$Tag launch ($mode, cap=$IterationCap, min=$RunMinimum) <- $Parent"
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

    $requestedStop = $false
    $lastIteration = -1
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 120
        $process.Refresh()
        $score = Get-Score $Tag $RunMinimum
        if ($score.current_iter -ne $null -and [int]$score.current_iter -ne $lastIteration) {
            $lastIteration = [int]$score.current_iter
            if ($score.ready) {
                Write-Status ("{0} iter={1} score={2:N3} survival={3:N3} clear={4:N3} exposure={5:N3} plateau={6}" -f $Tag, $score.current_iter, $score.current.score, $score.current.survival, $score.current.threat_clear, $score.current.exposure_clear, $score.plateau_age)
                if (
                    [int]$score.current_iter -ge $RunMinimum -and
                    [bool]$score.quality_ok -and
                    [int]$score.plateau_age -ge $PlateauPatience
                ) {
                    $requestedStop = $true
                    Write-Status "$Tag qualified and plateaued; requesting graceful stop at iter=$($score.current_iter)"
                    New-Item -ItemType File -Force -Path $stopFile | Out-Null
                    break
                }
            }
        }
    }
    Wait-Process -Id $process.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 20
    $final = Get-Score $Tag $RunMinimum
    $qualified = $false
    if ($final.ready) {
        $qualified = [bool]$final.quality_ok -and [int]$final.current_iter -ge $RunMinimum
        $summaryPath = Join-Path $LogRoot "$Tag\C_stage_summary.json"
        $final | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    }
    return [pscustomobject]@{
        Tag = $Tag
        Final = $final
        Qualified = $qualified
        RequestedStop = $requestedStop
    }
}

function Watch-AdoptedTrainingRun(
    [string]$Tag,
    [int]$ProcessId,
    [int]$RunMinimum
) {
    Write-Status "$Tag adopting existing run pid=$ProcessId"
    $process = Get-Process -Id $ProcessId -ErrorAction Stop
    $lastIteration = -1
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 120
        $process.Refresh()
        $score = Get-Score $Tag $RunMinimum
        if ($score.current_iter -ne $null -and [int]$score.current_iter -ne $lastIteration) {
            $lastIteration = [int]$score.current_iter
            if ($score.ready) {
                Write-Status ("{0} adopted iter={1} score={2:N3} survival={3:N3} clear={4:N3} exposure={5:N3}" -f $Tag, $score.current_iter, $score.current.score, $score.current.survival, $score.current.threat_clear, $score.current.exposure_clear)
            }
        }
    }
    Start-Sleep -Seconds 20
    $final = Get-Score $Tag $RunMinimum
    $qualified = $false
    if ($final.ready) {
        $qualified = [bool]$final.quality_ok -and [int]$final.current_iter -ge $RunMinimum
        $summaryPath = Join-Path $LogRoot "$Tag\C_stage_summary.json"
        $final | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    }
    return [pscustomobject]@{
        Tag = $Tag
        Final = $final
        Qualified = $qualified
        RequestedStop = $false
    }
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Write-Status "C-series start: stages=$($Stages -join ',') min=$MinIterations b-min=$ContinuationMinIterations plateau=$PlateauPatience threshold=$ScoreThreshold parent=$InitialBundle"
$parent = $InitialBundle

foreach ($stage in $Stages) {
    $tag = "$($stage)_crossdef"
    $experiment = Join-Path $Root "experiments\team01_$tag.yaml"
    if (-not (Test-Path -LiteralPath $experiment)) { throw "Experiment missing: $experiment" }

    if ($stage -eq "C1" -and $AdoptC1Pid -gt 0) {
        $run = Watch-AdoptedTrainingRun $tag $AdoptC1Pid $MinIterations
        $AdoptC1Pid = 0
    } else {
        $run = Invoke-CTrainingRun $tag $experiment $parent $false $MinIterations 300
    }
    if (-not $run.Qualified) {
        $checkpoint = Get-LastCheckpoint $tag
        if (-not $checkpoint) {
            Write-Status "$tag failed and has no checkpoint for $($stage)b; chain halted"
            break
        }
        $btag = "$($tag)b"
        Write-Status "$tag below gate; starting one full-state continuation as $btag"
        $run = Invoke-CTrainingRun $btag $experiment $checkpoint $true $ContinuationMinIterations 220
    }

    if (-not $run.Qualified) {
        $scoreText = if ($run.Final.ready) { "{0:N3}" -f $run.Final.current.score } else { "n/a" }
        Write-Status "$($run.Tag) failed after one continuation (score=$scoreText); chain halted"
        break
    }

    $parent = Get-PeakBundle $run.Tag ([int]$run.Final.best.iter)
    if (-not $parent) {
        Write-Status "$($run.Tag) has no peak bundle; chain halted"
        break
    }
    Write-Status "$($run.Tag) promoted from peak iter=$($run.Final.best.iter) score=$($run.Final.best.score) bundle=$parent"
}

Write-Status "C-series orchestrator finished; final parent=$parent"
