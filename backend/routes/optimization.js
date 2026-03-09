import { Router } from "express";
import { predictCost } from "../../server/controllers/mlController.js";
import { getConsolidationDashboard } from "../../server/controllers/consolidationController.js";

const router = Router();

router.post("/optimize-route", predictCost);
router.get("/consolidation/dashboard", getConsolidationDashboard);
router.get("/consolidation-dashboard", getConsolidationDashboard);

export default router;
