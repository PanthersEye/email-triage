"use client";
import { useEffect, useState } from "react";

type Email = { id: number; subject: string; body: string; created_at: string };
type Prediction = {
  id: number;
  email_id: number;
  category: string;
  probabilities: Record<string, number>;
  priority_score: number;
  priority_label: string;
  created_at: string;
};

export default function Detail({ params }: { params: { id: string } }) {
  const { id } = params;
  const [email, setEmail] = useState<Email | null>(null);
  const [preds, setPreds] = useState<Prediction[]>([]);

  useEffect(() => {
    const BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/,"");
    fetch(`${BASE}/emails/${id}`).then(r=>r.json()).then(setEmail);
    fetch(`${BASE}/emails/${id}/predictions`).then(r=>r.json()).then(setPreds);
  }, [id]);

  if (!email) return <main className="p-6">Loading...</main>;
  return (
    <main className="p-6 max-w-3xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">{email.subject || "(no subject)"}</h1>
      <pre className="whitespace-pre-wrap p-4 rounded-xl border">{email.body}</pre>
      <div className="space-y-2">
        {preds.map((p) => (
          <div key={p.id} className="p-3 rounded-xl border">
            <div>Category: <b>{p.category}</b></div>
            <div>Priority: <b>{p.priority_label}</b> ({Number(p.priority_score).toFixed(2)})</div>
          </div>
        ))}
      </div>
    </main>
  );
}
