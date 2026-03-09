import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createClient } from "@supabase/supabase-js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..");
const envPath = path.join(projectRoot, ".env");

function readDotEnvVar(key) {
  try {
    if (!fs.existsSync(envPath)) return undefined;
    const content = fs.readFileSync(envPath, "utf-8");
    const line = content
      .split(/\r?\n/)
      .find((item) => item.trim().startsWith(`${key}=`));
    return line ? line.split("=").slice(1).join("=").trim() : undefined;
  } catch (_error) {
    return undefined;
  }
}

const supabaseUrl = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL || readDotEnvVar("VITE_SUPABASE_URL");
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY || readDotEnvVar("VITE_SUPABASE_ANON_KEY");
const supabase = supabaseUrl && supabaseAnonKey ? createClient(supabaseUrl, supabaseAnonKey) : null;

const CLUSTER_COLORS = ["#2563eb", "#16a34a", "#db2777", "#ea580c", "#7c3aed", "#0f766e", "#9333ea"];

function isMissingResourceError(error) {
  if (!error) return false;
  if (error.status === 404) return true;
  if (error.code === "42P01" || error.code === "PGRST205") return true;
  const text = String(error.message || "").toLowerCase();
  return (
    text.includes("relation") ||
    text.includes("does not exist") ||
    text.includes("could not find the table")
  );
}

function isSupabaseUnavailableError(error) {
  if (!error) return false;
  const text = String(error.message || "").toLowerCase();
  return text.includes("supabase is not configured");
}

function isUpstreamFetchError(error) {
  if (!error) return false;
  const text = String(error.message || "").toLowerCase();
  return (
    text.includes("fetch failed") ||
    text.includes("network") ||
    text.includes("failed to fetch") ||
    text.includes("econnrefused") ||
    text.includes("enotfound") ||
    text.includes("timed out")
  );
}

function isPermissionError(error) {
  if (!error) return false;
  if (error.status === 401 || error.status === 403) return true;
  const text = String(error.message || "").toLowerCase();
  return (
    text.includes("permission denied") ||
    text.includes("row-level security") ||
    text.includes("not authorized") ||
    text.includes("jwt") ||
    text.includes("invalid api key") ||
    text.includes("forbidden")
  );
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function deadlineToMs(value) {
  if (!value) return null;
  const ts = new Date(value).getTime();
  return Number.isFinite(ts) ? ts : null;
}

function getRemainingMinutes(deadline) {
  const ms = deadlineToMs(deadline);
  if (!ms) return null;
  return Math.max(0, Math.round((ms - Date.now()) / 60000));
}

function normalizeOrder(row) {
  if (!row || typeof row !== "object") {
    return {
      id: null,
      order_id: null,
      pickup_latitude: 0,
      pickup_longitude: 0,
      delivery_latitude: 0,
      delivery_longitude: 0,
      pickup_area: "Unknown",
      packages: 1,
      weight: 0,
      volume: 0,
      distance_km: 0,
      load_type: "General",
      priority: "Medium",
      status: "available",
      delivery_deadline: null,
    };
  }
  return {
    id: row.id,
    order_id: row.order_id || row.id,
    pickup_latitude: toNumber(row.pickup_latitude ?? row.pickup_lat),
    pickup_longitude: toNumber(row.pickup_longitude ?? row.pickup_lng),
    delivery_latitude: toNumber(row.delivery_latitude ?? row.drop_lat),
    delivery_longitude: toNumber(row.delivery_longitude ?? row.drop_lng),
    pickup_area: row.pickup_area || row.pickup_address || "Unknown",
    packages: toNumber(row.packages ?? row.quantity, 1),
    weight: toNumber(row.weight),
    volume: toNumber(row.volume),
    distance_km: toNumber(row.distance_km ?? row.distance),
    load_type: row.load_type || row.commodity || "General",
    priority: row.priority || "Medium",
    status: row.status || "available",
    delivery_deadline: row.delivery_deadline || row.deadline || null,
  };
}

function isUrgentOrder(order) {
  const priority = String(order.priority || "").toLowerCase();
  const remaining = getRemainingMinutes(order.delivery_deadline);
  return priority === "high" || (remaining !== null && remaining <= 120);
}

function estimateVehicleCapacity(driver) {
  const type = String(driver.vehicle_type || "").toLowerCase();
  if (type.includes("heavy")) return { weight: 5000, volume: 25 };
  if (type.includes("mini")) return { weight: 1200, volume: 8 };
  return { weight: 500, volume: 3 };
}

function scoreCluster(orders, drivers) {
  if (orders.length === 0) return 0;
  const latAvg = orders.reduce((sum, item) => sum + item.delivery_latitude, 0) / orders.length;
  const lngAvg = orders.reduce((sum, item) => sum + item.delivery_longitude, 0) / orders.length;
  const maxSpread = orders.reduce((max, item) => {
    const spread = Math.abs(item.delivery_latitude - latAvg) + Math.abs(item.delivery_longitude - lngAvg);
    return Math.max(max, spread);
  }, 0);
  const proximityScore = Math.max(0, 1 - maxSpread * 40);

  const deadlines = orders
    .map((item) => deadlineToMs(item.delivery_deadline))
    .filter((item) => item !== null)
    .sort((a, b) => a - b);
  let windowScore = 0.7;
  if (deadlines.length > 1) {
    const spreadHours = (deadlines[deadlines.length - 1] - deadlines[0]) / 3600000;
    windowScore = Math.max(0, 1 - spreadHours / 12);
  }

  const totalWeight = orders.reduce((sum, item) => sum + item.weight, 0);
  const totalVolume = orders.reduce((sum, item) => sum + item.volume, 0);
  const capacities = drivers.map(estimateVehicleCapacity);
  const fitsAny = capacities.some((cap) => cap.weight >= totalWeight && cap.volume >= totalVolume);
  const capacityScore = fitsAny ? 1 : 0.45;

  const score = proximityScore * 0.45 + windowScore * 0.3 + capacityScore * 0.25;
  return Number((score * 100).toFixed(1));
}

function clusterOrders(orders, drivers) {
  const urgentOrders = orders.filter((order) => isUrgentOrder(order));
  const nonUrgentOrders = orders.filter((order) => !isUrgentOrder(order));
  const bucketMap = new Map();

  nonUrgentOrders.forEach((order) => {
    const area = order.pickup_area || "Unknown";
    const latBin = order.delivery_latitude.toFixed(2);
    const lngBin = order.delivery_longitude.toFixed(2);
    const dateBin = (order.delivery_deadline || "").slice(0, 10) || "none";
    const key = `${area}-${latBin}-${lngBin}-${dateBin}`;
    if (!bucketMap.has(key)) {
      bucketMap.set(key, {
        area,
        orders: [],
      });
    }
    bucketMap.get(key).orders.push(order);
  });

  const clusters = [...bucketMap.values()].map((cluster, index) => {
    const totalPackages = cluster.orders.reduce((sum, item) => sum + item.packages, 0);
    const totalWeight = cluster.orders.reduce((sum, item) => sum + item.weight, 0);
    const totalVolume = cluster.orders.reduce((sum, item) => sum + item.volume, 0);
    const averageDistance = cluster.orders.length
      ? cluster.orders.reduce((sum, item) => sum + item.distance_km, 0) / cluster.orders.length
      : 0;
    const centerLat = cluster.orders.reduce((sum, item) => sum + item.delivery_latitude, 0) / cluster.orders.length;
    const centerLng = cluster.orders.reduce((sum, item) => sum + item.delivery_longitude, 0) / cluster.orders.length;
    const consolidationScore = scoreCluster(cluster.orders, drivers);

    return {
      cluster_id: index + 1,
      area: cluster.area,
      shipments: cluster.orders.length,
      total_packages: Number(totalPackages.toFixed(2)),
      total_weight: Number(totalWeight.toFixed(2)),
      total_volume: Number(totalVolume.toFixed(2)),
      average_distance: Number(averageDistance.toFixed(2)),
      consolidation_score: consolidationScore,
      color: CLUSTER_COLORS[index % CLUSTER_COLORS.length],
      center_latitude: Number(centerLat.toFixed(6)),
      center_longitude: Number(centerLng.toFixed(6)),
      orders: cluster.orders,
    };
  });

  return { clusters, urgentOrders };
}

async function fetchFromFirstTable(tableNames, queryBuilder) {
  if (!supabase) return { data: [], error: new Error("Supabase is not configured on backend.") };

  let firstNonMissingError = null;
  let lastError = null;
  for (const tableName of tableNames) {
    const { data, error } = await queryBuilder(supabase.from(tableName));
    if (!error) return { data: data || [], error: null };
    lastError = error;
    if (!isMissingResourceError(error) && !firstNonMissingError) firstNonMissingError = error;
  }
  return { data: [], error: firstNonMissingError || lastError || new Error(`Could not fetch from tables: ${tableNames.join(", ")}`) };
}

export async function getConsolidationDashboard(req, res) {
  try {
    const [ordersResp, driversResp] = await Promise.all([
      fetchFromFirstTable(["orders", "Orders"], (table) => table.select("*")),
      fetchFromFirstTable(["drivers", "Driver"], (table) => table.select("*")),
    ]);

    const canFallbackToEmptyOrders =
      isMissingResourceError(ordersResp.error) ||
      isSupabaseUnavailableError(ordersResp.error) ||
      isUpstreamFetchError(ordersResp.error) ||
      isPermissionError(ordersResp.error);
    if (ordersResp.error && !canFallbackToEmptyOrders) {
      return res.status(500).json({ error: ordersResp.error.message });
    }

    const orderRows = Array.isArray(ordersResp.data) ? ordersResp.data : [];
    const orders = (ordersResp.error ? [] : orderRows)
      .filter((row) => row && typeof row === "object")
      .map(normalizeOrder);
    const canFallbackToEmptyDrivers =
      isMissingResourceError(driversResp.error) ||
      isSupabaseUnavailableError(driversResp.error) ||
      isUpstreamFetchError(driversResp.error) ||
      isPermissionError(driversResp.error);
    const driverRows = Array.isArray(driversResp.data) ? driversResp.data : [];
    const drivers = (driversResp.error && !canFallbackToEmptyDrivers ? [] : driverRows)
      .filter((driver) => driver && typeof driver === "object");
    const activeDrivers = drivers.filter((driver) => {
      const state = String(driver.status || "").toLowerCase();
      return ["active", "on trip", "in_transit", "in transit", "enroute", "assigned", "in_progress"].includes(state);
    });

    const { clusters, urgentOrders } = clusterOrders(orders, drivers);
    const shipmentsInClusters = clusters.reduce((sum, cluster) => sum + cluster.shipments, 0);
    const tripsWithout = orders.length;
    const tripsAfter = clusters.length + urgentOrders.length;
    const reduction = tripsWithout > 0 ? ((tripsWithout - tripsAfter) / tripsWithout) * 100 : 0;

    const clusterTable = clusters.map((cluster) => ({
      cluster_id: cluster.cluster_id,
      primary_delivery_area: cluster.area,
      shipments: cluster.shipments,
      total_packages: cluster.total_packages,
      total_weight: cluster.total_weight,
      total_volume: cluster.total_volume,
      average_distance: cluster.average_distance,
      consolidation_score: cluster.consolidation_score,
    }));

    const clusterDetails = clusters.reduce((acc, cluster) => {
      acc[cluster.cluster_id] = cluster.orders.map((order) => ({
        order_id: order.order_id,
        pickup_area: order.pickup_area,
        packages: order.packages,
        weight: order.weight,
        volume: order.volume,
        priority: order.priority,
        delivery_deadline: order.delivery_deadline,
        status: order.status,
      }));
      return acc;
    }, {});

    const mapZones = clusters.map((cluster) => ({
      cluster_id: cluster.cluster_id,
      area: cluster.area,
      color: cluster.color,
      center_latitude: cluster.center_latitude,
      center_longitude: cluster.center_longitude,
      shipment_points: cluster.orders.map((order) => ({
        order_id: order.order_id,
        latitude: order.delivery_latitude,
        longitude: order.delivery_longitude,
      })),
    }));

    const urgentShipments = urgentOrders.map((order) => ({
      order_id: order.order_id,
      pickup_area: order.pickup_area,
      priority: order.priority,
      delivery_deadline: order.delivery_deadline,
      remaining_time_minutes: getRemainingMinutes(order.delivery_deadline),
      reason_for_direct_dispatch: "High priority or delivery deadline too close",
    }));

    const averageShipmentsPerBatch = clusters.length ? shipmentsInClusters / clusters.length : 0;
    const averageWeightPerBatch = clusters.length
      ? clusters.reduce((sum, cluster) => sum + cluster.total_weight, 0) / clusters.length
      : 0;
    const averageVolumePerBatch = clusters.length
      ? clusters.reduce((sum, cluster) => sum + cluster.total_volume, 0) / clusters.length
      : 0;

    const bestArea = [...clusters].sort((a, b) => b.consolidation_score - a.consolidation_score)[0];
    const insights = [
      `${orders.length} total shipments analyzed for consolidation.`,
      `${clusters.length} optimized delivery batches created from non-urgent orders.`,
      `${urgentOrders.length} urgent shipment(s) bypassed consolidation for direct dispatch.`,
    ];
    if (bestArea) {
      insights.push(`Highest efficiency zone: ${bestArea.area} with consolidation score ${bestArea.consolidation_score}%.`);
    }

    const routeMatching = activeDrivers.slice(0, 5).map((driver, index) => ({
      id: `${driver.id || index}`,
      driver_name: driver.driver_name || driver.name || "Driver",
      vehicle_id: driver.vehicle_id || "N/A",
      route_area: driver.last_known_area || "Delivery Zone",
      action: "Add compatible nearby shipment to existing route",
    }));

    return res.json({
      summary: {
        total_shipments: orders.length,
        optimized_delivery_batches: clusters.length,
        trips_without_consolidation: tripsWithout,
        trips_after_consolidation: tripsAfter,
        trip_reduction_percentage: Number(reduction.toFixed(1)),
      },
      consolidation_efficiency: {
        trip_reduction_percentage: Number(reduction.toFixed(1)),
      },
      area_groups: clusterTable,
      cluster_table: clusterTable,
      cluster_details: clusterDetails,
      map_zones: mapZones,
      load_aggregation: clusters.map((cluster) => ({
        cluster_id: cluster.cluster_id,
        area: cluster.area,
        shipments: cluster.shipments,
        total_weight: cluster.total_weight,
        total_volume: cluster.total_volume,
        total_packages: cluster.total_packages,
      })),
      urgent_shipments: urgentShipments,
      analytics: {
        total_shipments: orders.length,
        optimized_delivery_batches: clusters.length,
        average_shipments_per_batch: Number(averageShipmentsPerBatch.toFixed(2)),
        average_weight_per_batch: Number(averageWeightPerBatch.toFixed(2)),
        average_volume_per_batch: Number(averageVolumePerBatch.toFixed(2)),
        cluster_density: Number(averageShipmentsPerBatch.toFixed(2)),
      },
      trip_reduction: {
        total_shipments: orders.length,
        trips_without_consolidation: tripsWithout,
        trips_after_consolidation: tripsAfter,
        trip_reduction_percentage: Number(reduction.toFixed(1)),
      },
      insights,
      route_matching: routeMatching,
      generated_at: new Date().toISOString(),
    });
  } catch (error) {
    return res.status(500).json({ error: error.message || "Failed to compute consolidation dashboard." });
  }
}

export const __private__ = {
  isMissingResourceError,
  isSupabaseUnavailableError,
  isUpstreamFetchError,
  isPermissionError,
  normalizeOrder,
};
