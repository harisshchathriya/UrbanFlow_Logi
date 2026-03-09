import { Router } from "express";

const router = Router();

router.get("/", async (_req, res) => {
  res.json({
    orders: [],
    message: "Orders API ready. Connect Supabase-backed order listing here.",
  });
});

router.post("/", async (req, res) => {
  res.status(201).json({
    order: req.body || {},
    message: "Order API ready. Persist this payload through Supabase in production.",
  });
});

export default router;
