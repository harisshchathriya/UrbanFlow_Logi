import { Router } from "express";

const router = Router();

router.get("/", async (_req, res) => {
  res.json({
    drivers: [],
    message: "Drivers API ready. Connect Supabase-backed driver listing here.",
  });
});

export default router;
