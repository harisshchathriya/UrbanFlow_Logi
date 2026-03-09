import express from "express";
import { corsMiddleware } from "./middleware/cors.js";
import mlRoutes from "./routes/mlRoutes.js";

const app = express();

app.use(corsMiddleware);
app.use(express.json());
app.use("/api", mlRoutes);

export default app;
