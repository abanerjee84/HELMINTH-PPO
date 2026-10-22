import { defineConfig } from 'vite';

// A relative base keeps the built bundle portable: it works from a GitHub
// Pages sub-path, from any static host, and straight off the filesystem.
export default defineConfig({
  base: './',
  build: {
    target: 'es2020',
  },
});
