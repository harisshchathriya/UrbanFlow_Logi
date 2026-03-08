# UrbanFlow

UrbanFlow is a React + Vite logistics platform with driver/operator dashboards and API routes for ML/consolidation endpoints.

## Local Development

1. Install dependencies:
   - `npm install`
2. Create local env:
   - copy `.env.example` to `.env`
   - fill required vars
3. Run frontend:
   - `npm run dev`
4. Optional backend (local Express):
   - `npm run server`

## Vercel Deployment

This repo is prepared for Vercel with:
- SPA routing rewrite (`vercel.json`)
- Serverless API routes:
  - `api/predict-cost.js`
  - `api/consolidation/dashboard.js`

### Vercel Project Settings

- Framework Preset: `Vite`
- Build Command: `npm run build`
- Output Directory: `dist`
- Install Command: `npm install`

### Required Environment Variables (Vercel)

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`
- `VITE_GOOGLE_MAPS_API_KEY`

Optional:
- `VITE_GOOGLE_MAP_ID`
- `SUPABASE_URL` (backend override)
- `SUPABASE_ANON_KEY` (backend override)
- `PYTHON_BIN` (only if using Python-backed prediction endpoint)
- `ML_FALLBACK_ONLY` (`1` to always use JS fallback estimator for `/api/predict-cost`)
- `ENABLE_PYTHON_ON_VERCEL` (`1` to try Python on Vercel; default is JS fallback)

## Notes

- Frontend calls `/api/*` are served by Vercel serverless functions under the `api/` folder.
- Client-side routes are rewritten to `index.html`, so direct navigation to nested routes works.
- On Vercel, `/api/predict-cost` now uses a JS fallback estimator by default to avoid Python runtime failures.
