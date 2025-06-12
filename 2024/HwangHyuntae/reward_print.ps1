# PowerShell 스크립트: run_rewards.ps1
# $eps_list = @(0, 5000, 10000, 15000, 20000, 25000, 30000)

# wltp
for ($eps = 1000; $eps -le 30000; $eps += 1000) {
    Write-Host "=== Testing episode $eps ==="
    # python .\visualize_train_result.py --test wltp --eps $eps --only-reward
    python .\verify.py --test wltp --eps $eps
}

# # udds
# for ($eps = 1000; $eps -le 31000; $eps += 1000) {
#     Write-Host "=== Testing episode $eps ==="
#     # python .\visualize_train_result.py --test wltp --eps $eps --only-reward
#     python .\verify.py --test udds --eps $eps
# }