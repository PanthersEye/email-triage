"use client";
import { useMemo, useState } from "react";

type PredictResponse = {
  email_id: number;
  category: string;
  probabilities: Record<string, number>;
  priority_score: number;
  priority_label: string;
};
type ReplyResponse = {
  email_id: number;
  draft: string;
  category: string;
  priority_label: string;
};
type ApiResult = PredictResponse | ReplyResponse;

export default function Compose() {
  const [subject, setSubject] = useState<string>("");
  const [body, setBody] = useState<string>("");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const BASE = useMemo(
    () => (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/,""),
    []
  );

  const post = (path: "predict" | "reply") => async () => {
    setErr(null); setResult(null);
    try {
      const r = await fetch(`${BASE}/${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ subject, body }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
      const data: ApiResult = await r.json();
      setResult(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <main className="p-6 max-w-3xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">Test an Email</h1>
      <div className="text-xs opacity-60">API: {BASE}</div>
      <input value={subject} onChange={e=>setSubject(e.target.value)} placeholder="Subject" className="w-full p-3 rounded-xl border"/>
      <textarea value={body} onChange={e=>setBody(e.target.value)} placeholder="Body" rows={10} className="w-full p-3 rounded-xl border"/>
      <div className="flex gap-2">
        <button onClick={post("predict")} className="px-4 py-2 rounded-2xl shadow">Predict</button>
        <button onClick={post("reply")} className="px-4 py-2 rounded-2xl shadow">Draft Reply</button>
      </div>
      {err && <div className="p-3 rounded-xl border border-red-400 text-red-700 whitespace-pre-wrap">Error: {err}</div>}
      {result && (<pre className="p-4 rounded-2xl border whitespace-pre-wrap">{JSON.stringify(result,null,2)}</pre>)}
    </main>
  );
}
