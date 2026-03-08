import { getConsolidationDashboard } from '../../server/controllers/consolidationController.js'

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET')
    return res.status(405).json({ error: 'Method Not Allowed' })
  }
  return getConsolidationDashboard(req, res)
}

