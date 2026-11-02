import { expect, test } from '@playwright/test';

test('schema form validates through the real backend and exports the current native spec', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', {name: 'LIF + Fire'})).toBeVisible();
  for (const [name, value] of Object.entries({tau: '0', threshold: '-50', v_reset: '-65', v_rest: '-65'})) {
    await page.getByLabel(name, {exact: true}).fill(value);
  }
  await page.getByRole('button', {name: 'Validate spec'}).click();
  await expect(page.getByLabel('tau', {exact: true})).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByText(/tau must be positive/)).toBeVisible();
  await page.getByLabel('tau', {exact: true}).fill('20');
  await page.getByRole('button', {name: 'Validate spec'}).click();
  await expect(page.getByRole('status')).toContainText('Spec validated');
  await page.getByRole('button', {name: 'Your network'}).click();
  await expect(page.locator('pre')).toContainText('"tau": 20');
});

test('reduced motion has no running animations and the form fits mobile', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('/');
  await expect(page.getByLabel('tau', {exact: true})).toBeVisible();
  expect(await page.evaluate(() => document.getAnimations().filter(a => a.playState === 'running').length)).toBe(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', {name: 'Reduce motion'}).click();
  await page.reload();
  await expect(page.getByRole('button', {name: 'Reduce motion'})).toHaveAttribute('aria-pressed', 'true');
});
