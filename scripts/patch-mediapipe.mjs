#!/usr/bin/env node
// Patch @mediapipe/tasks-vision/package.json to fix invalid "exports" field
// that mixes condition keys (import/require/default/types) with subpath keys (./).
// rolldown (used by Vite 8) strictly rejects such mixed exports objects.
import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const pkgPath = resolve(__dirname, '..', 'node_modules/@mediapipe/tasks-vision/package.json');
const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));

const exports_ = pkg.exports;
if (!exports_ || typeof exports_ !== 'object') process.exit(0);

const keys = Object.keys(exports_);
const hasSubpaths = keys.some(k => k.startsWith('./'));
const hasConditions = keys.some(k => !k.startsWith('.'));

if (hasSubpaths && hasConditions) {
  // Move top-level condition keys under "."
  const dotEntry = exports_['.'] ?? {};
  const newExports = { '.': { ...dotEntry } };
  for (const [k, v] of Object.entries(exports_)) {
    if (k.startsWith('.') && k !== '.') {
      newExports[k] = v;
    } else if (k !== '.') {
      // condition key (import/require/default/types) → merge into "."
      newExports['.'][k] = v;
    }
  }
  pkg.exports = newExports;
  writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  console.log('Patched @mediapipe/tasks-vision package.json exports.');
} else {
  console.log('@mediapipe/tasks-vision exports OK, no patch needed.');
}
