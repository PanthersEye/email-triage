"use client";
import { useEffect, useState } from "react";
export default function Detail({ params }: any) {
  const { id } = params;
  const [email, setEmail] = useState<any>(null);
  const [preds, setPreds] = useState<any[]>([]);
  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_BASE}/emails/${id}`).then(r=>r.json()).then(setEmail);
    fetch(`${process.env.NEXT_PUBLIC_API_BASE}/emails/${id}/predictions`).then(r=>r.json()).then(setPreds);
  }, [id]);
  if (!email) return <main className="p-6">Loading...</main>;
  return (
    <main className="p-6 max-w-3xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">{email.subject || "(no subject)"}</h1>
      <pre className="whitespace-pre-wrap p-4 rounded-xl border">{email.body}</pre>
      <div className="space-y-2">
        {preds.map((p:any) => (
          <div key={p.id} className="p-3 rounded-xl border">
            <div>Category: <b>{p.category}</b></div>
            <div>Priority: <b>{p.priority_label}</b> ({Number(p.priority_score).toFixed?.(2)})</div>
          </div>
        ))}
      </div>
    </main>
  );
}
