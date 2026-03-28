/**
 * Coral TPU Monitor
 * Host-side wrapper to launch the Coral TPU Docker container.
 * Matches the CameraClaw architecture by acting as the skill entrypoint.
 */

const { spawn } = require('node:child_process');
const os = require('node:os');
const fs = require('node:fs');

function main() {
  const imageName = 'aegis-coral-tpu';
  const imageTag = 'latest';

  const cmd = 'docker';
  const args = ['run', '-i', '--rm'];

  // Extra PATH augmentation to ensure docker is found when launched via Electron on macOS
  const extraPaths = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin'];
  const currentPath = process.env.PATH || '';
  const missing = extraPaths.filter(p => !currentPath.split(':').includes(p));
  if (missing.length > 0) {
    process.env.PATH = [...missing, currentPath].join(':');
  }

  // Handle USB Passthrough (Coral Edge TPU)
  // macOS/Windows handle USB dynamically via Docker Desktop 4.35+
  // Only Linux requires explicit device mounting
  if (os.platform() === 'linux' && fs.existsSync('/dev/bus/usb')) {
    args.push('--device', '/dev/bus/usb:/dev/bus/usb');
  }

  // Shared memory volume for video frames
  const path = require('node:path');
  const sharedMemoryHost = path.join(os.tmpdir(), 'aegis-detection-frames');
  if (!fs.existsSync(sharedMemoryHost)) {
    fs.mkdirSync(sharedMemoryHost, { recursive: true });
  }
  // Map the host path to the EXACT same absolute path inside the container
  // This allows the raw JSON `frame_path` from Aegis to work without translation.
  args.push('-v', `${sharedMemoryHost}:${sharedMemoryHost}`);

  // Pass through Aegis parameters and ID dynamically
  for (const [key, val] of Object.entries(process.env)) {
    if (key.startsWith('AEGIS_') || key === 'PYTHONUNBUFFERED') {
      args.push('--env', `${key}=${val}`);
    }
  }

  if (!process.env.PYTHONUNBUFFERED) {
    args.push('--env', 'PYTHONUNBUFFERED=1');
  }

  args.push(`${imageName}:${imageTag}`);

  const child = spawn(cmd, args, {
    stdio: 'inherit'
  });

  child.on('error', (err) => {
    console.log(JSON.stringify({
      event: 'error',
      message: `Docker is not installed or not in PATH: ${err.message}`,
      retriable: false
    }));
    process.exit(1);
  });

  child.on('exit', (code) => {
    process.exit(code || 0);
  });
}

main();
