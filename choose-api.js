#!/usr/bin/env node
/**
 * Foto Render API Selection
 * Prompts user to choose between monolithic or queue-based API
 */

const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log('\n🎨 FOTO RENDER API LAUNCHER');
console.log('='.repeat(40));
console.log('\nChoose your API mode:\n');
console.log('  [M] 🏠 Monolithic (main.py)');
console.log('      ✅ Steady power draw (no GPU spikes)');
console.log('      ✅ Simple and reliable');
console.log('      ❌ Single user only\n');
console.log('  [Q] ⚡ Queue-based (api_v2.py)');
console.log('      ✅ Multiple users');
console.log('      ✅ Non-blocking generation');
console.log('      ❌ Potential power spikes');
console.log('      ℹ️  Requires Redis\n');

function startAPI(mode) {
  const backendDir = path.join(__dirname, 'backend');
  
  let apiFile, description;
  if (mode.toLowerCase() === 'm') {
    apiFile = 'main.py';
    description = '🏠 MONOLITHIC API (no power spikes)';
  } else {
    apiFile = 'api_v2.py';
    description = '⚡ QUEUE-BASED API (multi-user)';
  }

  console.log(`\n🚀 Starting ${description}...\n`);

  // Use Python 3.13 directly with full path (bypass PATH issues)
  const pythonPath = process.platform === 'win32' ? 'C:\\Python313\\python.exe' : 'python3.13';
  const backend = spawn(pythonPath, [apiFile, '--port', '8000'], {
    cwd: backendDir,
    stdio: 'inherit'
  });

  backend.on('error', (err) => {
    console.error('❌ Failed to start backend:', err.message);
    process.exit(1);
  });

  backend.on('exit', (code) => {
    if (code !== 0) {
      console.error(`❌ Backend exited with code ${code}`);
      process.exit(code);
    }
  });

  // Handle Ctrl+C gracefully
  process.on('SIGINT', () => {
    console.log('\n👋 Shutting down backend...');
    backend.kill('SIGTERM');
    process.exit(0);
  });
}

rl.question('Enter your choice (M/Q): ', (answer) => {
  rl.close();
  
  const choice = answer.trim().toLowerCase();
  if (choice === 'm' || choice === 'q') {
    startAPI(choice);
  } else {
    console.log('\n❌ Invalid choice. Please run again and choose M or Q.');
    process.exit(1);
  }
}); 