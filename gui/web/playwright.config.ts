import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e', workers: 1,
  use: { channel: 'chromium', baseURL: 'http://127.0.0.1:8876' },
  webServer: {
    command: 'cd ../.. && gui/run-server --port 8876',
    url: 'http://127.0.0.1:8876/api/health', reuseExistingServer: false, timeout: 30000,
  },
});
