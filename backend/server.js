import cors from "cors";
import express from "express";

import driversRoutes from "./routes/drivers.js";
import healthRoutes from "./routes/health.js";
import optimizationRoutes from "./routes/optimization.js";
import ordersRoutes from "./routes/orders.js";

const app = express();

function isAllowedOrigin(origin) {
  if (!origin) return true;

  const allowedOrigins = [
    "http://localhost:5173",
    process.env.FRONTEND_URL,
    process.env.VERCEL_FRONTEND_URL,
  ].filter(Boolean);

  if (allowedOrigins.includes(origin)) return true;

  try {
    const hostname = new URL(origin).hostname;
    return hostname.endsWith(".vercel.app");
  } catch {
    return false;
  }
}

app.use(
  cors({
    origin(origin, callback) {
      if (isAllowedOrigin(origin)) {
        callback(null, true);
        return;
      }
      callback(new Error(`CORS blocked for origin: ${origin}`));
    },
    credentials: true,
  }),
);

app.use(express.json());

app.use("/api", healthRoutes);
app.use("/api/orders", ordersRoutes);
app.use("/api/drivers", driversRoutes);
app.use("/api", optimizationRoutes);

const PORT = process.env.PORT || 5000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});
