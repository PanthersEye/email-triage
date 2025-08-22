"use client";
import { useState } from "react";
export default function Compose() {
  const [subject, setSubject] = useState(""); const [body, setBody] = useState("");
  const [result, setResult] = useState<any>(null);
  const post = (path:string) => async () => {
    const r = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/${path}`, {
      method:"POST", headers:{"content-type":"application/json"},
      body: JSON.stringify({subject, body})
    });
    setResult(await r.json());
  };
  return (
    <main className="p-6 max-w-3xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">Test an Email</h1>
      <input value={subject} onChange={e=>setSubject(e.target.value)} placeholder="Subject" className="w-full p-3 rounded-xl border"/>
      <textarea value={body} onChange={e=>setBody(e.target.value)} placeholder="Body" rows={10} className="w-full p-3 rounded-xl border"/>
      <div className="flex gap-2">
        <button onClick={post("predict")} className="px-4 py-2 rounded-2xl shadow">Predict</button>
        <button onClick={post("reply")} className="px-4 py-2 rounded-2xl shadow">Draft Reply</button>
      </div>
      {result && (<pre className="p-4 rounded-2xl border whitespace-pre-wrap">{JSON.stringify(result,null,2)}</pre>)}
    </main>
  );
}
