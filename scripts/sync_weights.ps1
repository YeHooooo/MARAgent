param(
    [string]$SourceRoot = "D:\MAR\Agent3_VLM_more_model_different_part",
    [string]$TargetRoot = "D:\MAR\MARAgent_git"
)

$pairs = @(
    @("tools\DICDNet\pretrain_model\DICDNet_latest.pt", "tools\DICDNet\pretrain_model\DICDNet_latest.pt"),
    @("tools\OSCNet\pretrained_model\model_osc\net_latest.pt", "tools\OSCNet\pretrained_model\model_osc\net_latest.pt"),
    @("tools\OSCNet\pretrained_model\model_oscplus\net_latest.pt", "tools\OSCNet\pretrained_model\model_oscplus\net_latest.pt"),
    @("tools\InDuDoNet\pretrained_model\InDuDoNet_latest.pt", "tools\InDuDoNet\pretrained_model\InDuDoNet_latest.pt"),
    @("tools\InDuDoNet_plus\pretrained_model\InDuDoNet+_latest.pt", "tools\InDuDoNet_plus\pretrained_model\InDuDoNet+_latest.pt"),
    @("tools\ACDNet\models\ACDNet_latest.pt", "tools\ACDNet\models\ACDNet_latest.pt"),
    @("tools\adn\adn\runs\mmdental\net_199.pt", "tools\adn\adn\runs\mmdental\net_199.pt"),
    @("tools\SemiMAR\SemiMAR\runs\yofo_data\net_199.pt", "tools\SemiMAR\SemiMAR\runs\yofo_data\net_199.pt"),
    @("tools\calimar\checkpoints\my_calimar_training\latest_net_G_A.pth", "tools\calimar\checkpoints\my_calimar_training\latest_net_G_A.pth")
)

foreach ($pair in $pairs) {
    $src = Join-Path $SourceRoot $pair[0]
    $dst = Join-Path $TargetRoot $pair[1]
    if (-not (Test-Path $src)) {
        Write-Warning "Missing source checkpoint: $src"
        continue
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -Path $src -Destination $dst -Force
    Write-Host "Copied $src -> $dst"
}
