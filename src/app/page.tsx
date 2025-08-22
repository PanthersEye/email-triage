"use client";
import { useEffect, useState } from "react";
type Email = { id:number; subject:string; created_at:string };
export default function Home() {
  const [items, setItems] = useState<Email[]>([]);
  useEffect(() => { fetch(`${process.env.NEXT_PUBLIC_API_BASE}/emails`).then(r=>r.json()).then(setItems); }, []);
  return (
    <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Email Triage</h1>
      <a className="underline" href="/compose">+ New test email</a>
      <ul className="mt-6 space-y-3">
        {items.map(e => (
          <li key={e.id} className="p-4 rounded-2xl shadow flex justify-between">
            <div>
              <a className="font-medium" href={`/email/${e.id}`}>{e.subject || "(no subject)"}</a>
              <div className="text-sm opacity-70">{new Date(e.created_at).toLocaleString()}</div>
            </div>
          </li>
        ))}
      </ul>
    </main>
  );
}
