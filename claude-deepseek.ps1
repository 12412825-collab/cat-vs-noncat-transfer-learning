$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"

if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host "错误: 未设置 DeepSeek API Key!" -ForegroundColor Red
    Write-Host "请先在当前终端执行以下命令设置 API Key：" -ForegroundColor Yellow
    Write-Host "`$env:ANTHROPIC_API_KEY='sk-你的deepseek-api-key'" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Yellow
    exit 1
}

Write-Host "正在启动 Claude Code，已接入 DeepSeek API (https://api.deepseek.com/anthropic)..." -ForegroundColor Green
claude
