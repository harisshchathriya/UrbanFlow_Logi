import { __private__ } from "./consolidationController.js";

describe("consolidation controller helpers", () => {
  test("treats Supabase auth and RLS failures as permission errors", () => {
    expect(__private__.isPermissionError({ status: 401, message: "Invalid JWT" })).toBe(true);
    expect(__private__.isPermissionError({ status: 403, message: "new row violates row-level security policy" })).toBe(true);
    expect(__private__.isPermissionError({ message: "permission denied for table orders" })).toBe(true);
  });

  test("normalizes malformed order rows without throwing", () => {
    expect(__private__.normalizeOrder(null)).toEqual({
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
    });
  });
});
