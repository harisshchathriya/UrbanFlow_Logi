import { buildApiUrl } from '../config/api'

async function fetchApi(input, init) {
  try {
    return await fetch(input, init)
  } catch (error) {
    const text = String(error?.message || '').toLowerCase()
    if (text.includes('failed to fetch') || text.includes('networkerror') || text.includes('load failed')) {
      throw new Error('Backend API is unavailable. Start `npm.cmd run server` on port 5000.')
    }
    throw error
  }
}

async function parseApiResponse(response) {
  const raw = await response.text()
  let data = null
  try {
    data = raw ? JSON.parse(raw) : {}
  } catch (_error) {
    if (!response.ok) {
      if (response.status === 500 && !raw.trim()) {
        throw new Error('Backend API is unavailable. Start `npm.cmd run server` on port 5000.')
      }
      if (raw.trim().startsWith('<')) {
        throw new Error(`Backend request failed with HTTP ${response.status}. Check VITE_API_URL and backend logs.`)
      }
      throw new Error(raw?.slice(0, 180) || `HTTP ${response.status}`)
    }
    throw new Error('API returned non-JSON response.')
  }
  if (!response.ok) {
    throw new Error(data?.error || `HTTP ${response.status}`)
  }
  return data
}

export async function predictTripCost(orderData) {
  const response = await fetchApi(buildApiUrl('/api/predict-cost'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(orderData),
  })
  return parseApiResponse(response)
}

export async function fetchExampleTripCost() {
  const sampleOrder = {
    weight: 12,
    volume: 1.3,
    distance_km: 10,
    priority_num: 3,
    fuel_used: 0.5,
    time_remaining_hr: 4,
  }

  return predictTripCost(sampleOrder)
}

export async function fetchConsolidationDashboard() {
  const endpoints = [buildApiUrl('/api/consolidation/dashboard')]
  let lastError = null
  for (const endpoint of endpoints) {
    try {
      const response = await fetchApi(endpoint)
      return await parseApiResponse(response)
    } catch (error) {
      lastError = error instanceof Error
        ? error
        : new Error('Failed to fetch consolidation dashboard.')
    }
  }
  throw lastError || new Error('Failed to fetch consolidation dashboard.')
}
