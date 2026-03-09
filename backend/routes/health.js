import { Router } from "express";

const router = Router();

router.get("/health", (_req, res) => {
  res.json({ status: "UrbanFlow backend running" });
});

export default router;
