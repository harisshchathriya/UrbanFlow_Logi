import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import GlassCard from '../components/GlassCard'
import StatCard from '../components/StatCard'
import PageBackButton from '../components/PageBackButton'
import { fetchConsolidationDashboard } from '../api/mlApi'
import { loadGoogleMaps } from '../lib/maps'
import { getStoredOperator } from '../lib/session'

function formatNumber(value, suffix = '') {
  const num = Number(value)
  if (!Number.isFinite(num)) return `0${suffix}`
  return `${num}${suffix}`
}

export default function LoadConsolidation() {
  const navigate = useNavigate()
  const operator = useMemo(() => getStoredOperator(), [])
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [selectedClusterId, setSelectedClusterId] = useState(null)
  const [mapError, setMapError] = useState('')
  const mapRef = useRef(null)

  useEffect(() => {
    if (!operator?.email) {
      navigate('/operator-login')
      return undefined
    }
    let mounted = true
    let timer = null
    const load = async () => {
      try {
        const payload = await fetchConsolidationDashboard()
        if (mounted) {
          setData(payload)
          setError('')
        }
        return true
      } catch (e) {
        if (mounted) setError(String(e?.message || 'Failed to fetch ML dashboard'))
        return false
      }
    }

    load().then((ok) => {
      if (!mounted || !ok) return
      timer = setInterval(load, 15000)
    })
    return () => {
      mounted = false
      if (timer) clearInterval(timer)
    }
  }, [navigate, operator?.email])

  const summary = data?.summary || {}
  const analytics = data?.analytics || {}
  const tripReduction = data?.trip_reduction || {}
  const clusters = data?.cluster_table || []
  const urgent = data?.urgent_shipments || []
  const insights = data?.insights || []
  const routeMatching = data?.route_matching || []
  const clusterDetails = data?.cluster_details || {}
  const mapZones = data?.map_zones || []

  const legends = useMemo(() => {
    const zones = data?.map_zones || []
    return zones.slice(0, 3).map((item) => item.area)
  }, [data])

  useEffect(() => {
    const renderMap = async () => {
      if (!mapRef.current) return
      if (mapZones.length === 0) return
      try {
        const googleObj = await loadGoogleMaps(import.meta.env.VITE_GOOGLE_MAPS_API_KEY)
        const first = mapZones[0]
        const center = {
          lat: Number(first.center_latitude) || 13.0827,
          lng: Number(first.center_longitude) || 80.2707,
        }
        const map = new googleObj.maps.Map(mapRef.current, { center, zoom: 11 })
        const bounds = new googleObj.maps.LatLngBounds()

        mapZones.forEach((zone) => {
          const lat = Number(zone.center_latitude)
          const lng = Number(zone.center_longitude)
          if (Number.isFinite(lat) && Number.isFinite(lng)) {
            const position = { lat, lng }
            bounds.extend(position)
            new googleObj.maps.Marker({
              position,
              map,
              title: `Cluster ${zone.cluster_id}: ${zone.area}`,
            })
            new googleObj.maps.Circle({
              map,
              center: position,
              radius: 1200,
              strokeColor: zone.color || '#60a5fa',
              strokeOpacity: 0.8,
              strokeWeight: 2,
              fillColor: zone.color || '#60a5fa',
              fillOpacity: 0.2,
            })
          }

          ;(zone.shipment_points || []).forEach((point) => {
            const pLat = Number(point.latitude)
            const pLng = Number(point.longitude)
            if (!Number.isFinite(pLat) || !Number.isFinite(pLng)) return
            const pointPos = { lat: pLat, lng: pLng }
            bounds.extend(pointPos)
            new googleObj.maps.Marker({
              position: pointPos,
              map,
              title: `Order ${point.order_id}`,
              icon: {
                path: googleObj.maps.SymbolPath.CIRCLE,
                scale: 4,
                fillColor: zone.color || '#93c5fd',
                fillOpacity: 1,
                strokeWeight: 0,
              },
            })
          })
        })

        if (!bounds.isEmpty()) map.fitBounds(bounds)
        setMapError('')
      } catch (e) {
        setMapError(String(e?.message || 'Map failed to load'))
      }
    }

    renderMap()
  }, [mapZones])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-700 to-teal-500 p-6">
      <PageBackButton />
      <div className="max-w-7xl mx-auto space-y-4">
        <h1 className="text-white text-3xl font-semibold">ML-Based Load Consolidation</h1>

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <StatCard label="Total Orders" value={formatNumber(summary.total_shipments)} />
          <StatCard label="Consolidated Clusters" value={formatNumber(summary.optimized_delivery_batches)} />
          <StatCard label="Trips Without AI" value={formatNumber(summary.trips_without_consolidation)} />
          <StatCard label="Trips With AI" value={formatNumber(summary.trips_after_consolidation)} />
          <StatCard label="Trip Reduction %" value={formatNumber(summary.trip_reduction_percentage, '%')} />
        </div>

        <GlassCard className="p-5 overflow-auto">
          <h3 className="text-white text-xl font-semibold">Consolidation Clusters</h3>
          <table className="w-full mt-3 text-sm text-white/90">
            <thead><tr className="text-left text-white/70"><th>Cluster ID</th><th>Primary Area</th><th>Orders</th><th>Packages</th><th>Weight</th><th>Volume</th><th>Average Distance</th><th>Action</th></tr></thead>
            <tbody>
              {clusters.map((row) => (
                <tr key={row.cluster_id} className="border-t border-white/15">
                  <td>{row.cluster_id}</td><td>{row.primary_delivery_area}</td><td>{row.shipments}</td><td>{row.total_packages}</td><td>{row.total_weight} kg</td><td>{row.total_volume} m3</td><td>{row.average_distance} km</td>
                  <td><button className="bg-white/20 hover:bg-white/30 text-white rounded-xl px-3 py-1" onClick={() => setSelectedClusterId(String(row.cluster_id))}>View Details</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>

        {selectedClusterId ? (
          <GlassCard className="p-5 overflow-auto">
            <div className="flex items-center justify-between">
              <h3 className="text-white text-xl font-semibold">Cluster {selectedClusterId} Details</h3>
              <button className="bg-white/20 hover:bg-white/30 text-white rounded-xl px-3 py-1" onClick={() => setSelectedClusterId(null)}>Close</button>
            </div>
            <table className="w-full mt-3 text-sm text-white/90">
              <thead><tr className="text-left text-white/70"><th>Order ID</th><th>Pickup Area</th><th>Packages</th><th>Weight</th><th>Volume</th><th>Priority</th><th>Deadline</th><th>Status</th></tr></thead>
              <tbody>
                {(clusterDetails[selectedClusterId] || []).map((row) => (
                  <tr key={row.order_id} className="border-t border-white/15">
                    <td>{row.order_id}</td><td>{row.pickup_area}</td><td>{row.packages}</td><td>{row.weight}</td><td>{row.volume}</td><td>{row.priority}</td><td>{row.delivery_deadline || '-'}</td><td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </GlassCard>
        ) : null}

        <GlassCard className="p-6">
          <p className="text-white text-2xl font-semibold text-center">Cluster Visualization</p>
          <p className="text-white/75 text-center">Color-coded delivery zones</p>
          <div ref={mapRef} className="w-full h-[360px] rounded-xl mt-4 bg-white/10" />
          <p className="text-white/65 mt-3 text-center">Legend: {legends.length ? legends.join(' | ') : 'No cluster zones available'}</p>
          {mapError ? <p className="text-red-200 text-center mt-2">{mapError}</p> : null}
        </GlassCard>

        <GlassCard className="p-5">
          <h3 className="text-white text-xl font-semibold">Urgent Shipments</h3>
          <div className="mt-3 space-y-2 text-white/90">
            {urgent.map((item) => (
              <div key={item.order_id}>
                Order ID {item.order_id} | Pickup Area {item.pickup_area} | Priority {item.priority} | Deadline {item.delivery_deadline || '-'} | Remaining {item.remaining_time_minutes ?? '-'} min | Reason {item.reason_for_direct_dispatch}
              </div>
            ))}
            {urgent.length === 0 ? <div>No urgent shipments right now.</div> : null}
          </div>
        </GlassCard>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <StatCard label="Average Orders per Cluster" value={formatNumber(analytics.average_shipments_per_batch)} />
          <StatCard label="Average Weight per Cluster" value={formatNumber(analytics.average_weight_per_batch, ' kg')} />
          <StatCard label="Average Volume per Cluster" value={formatNumber(analytics.average_volume_per_batch, ' m3')} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <GlassCard className="p-5">
            <h3 className="text-white text-xl font-semibold">AI Generated Insights</h3>
            {insights.map((item, idx) => <p key={`${item}-${idx}`} className="text-white/85 mt-2">{item}</p>)}
            {insights.length === 0 ? <p className="text-white/85 mt-2">No insights available.</p> : null}
          </GlassCard>
          <GlassCard className="p-5">
            <h3 className="text-white text-xl font-semibold">Trip Reduction Impact</h3>
            <p className="text-white/85 mt-2">Before Consolidation: Total Shipments {tripReduction.total_shipments || 0}, Required Trips {tripReduction.trips_without_consolidation || 0}</p>
            <p className="text-white/85 mt-1">After ML Consolidation: Optimized Trips {tripReduction.trips_after_consolidation || 0}, Trips Saved {(tripReduction.trips_without_consolidation || 0) - (tripReduction.trips_after_consolidation || 0)}</p>
          </GlassCard>
        </div>

        <GlassCard className="p-5">
          <h3 className="text-white text-xl font-semibold">Route Matching Suggestions</h3>
          <div className="mt-3 space-y-2 text-white/90">
            {routeMatching.map((row) => (
              <div key={row.id}>Driver {row.driver_name} ({row.vehicle_id}) | Area {row.route_area} | Action: {row.action}</div>
            ))}
            {routeMatching.length === 0 ? <div>No live route matching recommendations.</div> : null}
          </div>
        </GlassCard>

        {error ? <p className="text-red-200">ML fetch error: {error}</p> : null}
      </div>
    </div>
  )
}
