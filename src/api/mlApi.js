async function parseApiResponse(response) {
  const raw = await response.text()
  let data = null
  try {
    data = raw ? JSON.parse(raw) : {}
  } catch (_error) {
    if (!response.ok) {
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
  const response = await fetch('/api/predict-cost', {
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
  const response = await fetch('/api/consolidation/dashboard')
  return parseApiResponse(response)
}

