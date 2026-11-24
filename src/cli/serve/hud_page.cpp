// The `/hud` page: the engine-native inference dashboard. Same layout as the
// standalone x3mon HUD, but it talks only to this process — `/hud/api/state`
// — and carries no external scrape source.
#include <string_view>

#include "src/cli/serve/hud_state.hpp"

namespace gufo::server {

std::string_view HudPage() {
  // Raw literals in pieces under MSVC's 16 KiB limit per literal.
  return R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gufo · LLM Inference</title>
<style>
:root{
  --bg:#0a0c0b; --card:#141716; --card2:#181c1a; --line:#242a27;
  --txt:#e9efeb; --dim:#8b948e; --faint:#5c6560;
  --green:#3ddc97; --teal:#2dd4bf; --blue:#4da3ff; --amber:#f5c04e; --red:#ff6b7a; --gray:#3a423e;
  --font:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--font);min-height:100vh;padding:18px 22px 34px}
.mono-num{font-variant-numeric:tabular-nums}

/* ---------- top bar ---------- */
.topbar{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin-bottom:16px}
.topbar .row1{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.topbar h1{font-size:15px;font-weight:700;letter-spacing:.4px}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--faint);
  border:1px solid var(--line);border-radius:99px;padding:4px 12px;background:var(--card2)}
.pill .dot{width:7px;height:7px;border-radius:50%;background:currentColor;opacity:.35}
.pill.on{color:var(--txt);border-color:#3a4641}
.pill.on .dot{opacity:1;box-shadow:0 0 8px currentColor}
.pill.c-green{color:var(--green)} .pill.c-blue{color:var(--blue)} .pill.c-amber{color:var(--amber)}
.pill.c-red{color:var(--red)} .pill.c-gray{color:#aab4ae} .pill.c-teal{color:var(--teal)}
.topbar .row2{display:flex;justify-content:space-between;margin-top:10px;font-size:13px;color:var(--dim)}
.topbar .row2 b{color:var(--txt);font-weight:600}

/* ---------- card grid ---------- */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;min-height:118px}
.card h2{font-size:12px;font-weight:600;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;
  display:flex;align-items:center;gap:8px;margin-bottom:10px}
.card h2 svg{width:14px;height:14px;stroke:var(--dim);fill:none;stroke-width:1.8}
.big{font-size:34px;font-weight:700;letter-spacing:-.5px}
.big .unit{font-size:16px;color:var(--dim);font-weight:500;margin-left:4px}
.sub{font-size:12.5px;color:var(--dim);margin-top:5px}
.duo{display:flex;gap:34px}
.duo>div{flex:1}
.kv{display:flex;justify-content:space-between;font-size:12.5px;color:var(--dim);padding:2.5px 0}
.kv b{color:var(--txt);font-weight:600}
.bar{height:5px;border-radius:99px;background:var(--gray);margin-top:12px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:var(--green);transition:width .6s}
.spark{margin-top:10px;width:100%;height:38px;display:block}
#sparkTip{display:none;position:fixed;z-index:50;background:rgba(20,22,26,.96);border:1px solid var(--line);border-radius:6px;padding:6px 9px;pointer-events:none;box-shadow:0 4px 14px rgba(0,0,0,.45);font-size:11px;line-height:1.5;white-space:nowrap}
#sparkTip .t1{font-weight:600}
#sparkTip .t2{display:flex;align-items:center;gap:6px;margin-top:2px}
#sparkTip .sw{width:8px;height:8px;border-radius:2px;display:inline-block}
#sparkTip .lb{color:var(--dim)}
#sparkTip .vl{font-weight:600}
.g{color:var(--green)}.t{color:var(--teal)}.b{color:var(--blue)}.a{color:var(--amber)}.r{color:var(--red)}

/* ---------- bottom row ---------- */
.bottom{display:grid;grid-template-columns:380px 1fr;gap:14px}
@media(max-width:980px){.grid,.bottom{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.panel h2{font-size:12px;font-weight:600;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.donut-wrap{display:flex;justify-content:center;padding:6px 0 2px}
.legend{display:flex;gap:14px;justify-content:center;font-size:12px;color:var(--dim);margin:10px 0 16px;flex-wrap:wrap}
.legend span{display:flex;align-items:center;gap:6px}
.legend i{width:9px;height:9px;border-radius:2px;display:inline-block}
.hbar{margin:9px 0}
.hbar .lbl{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-bottom:4px}
.hbar .lbl b{color:var(--txt)}
.hbar .trk{height:6px;background:var(--gray);border-radius:99px;overflow:hidden}
.hbar .trk i{display:block;height:100%;border-radius:99px;transition:width .6s}

/* ---------- table ---------- */
.btn{font-size:11.5px;color:var(--dim);background:var(--card2);border:1px solid var(--line);
  border-radius:8px;padding:4px 10px;cursor:pointer}
.btn:hover{color:var(--txt)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--faint);font-weight:600;text-transform:uppercase;font-size:10.5px;letter-spacing:.6px;
  text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:6.5px 8px;text-align:right;border-bottom:1px solid #1c211f;color:var(--txt)}
tr:last-child td{border-bottom:none}
.st{display:inline-flex;align-items:center;gap:6px;font-size:11.5px}
.st i{width:6px;height:6px;border-radius:50%;display:inline-block}
.empty{color:var(--faint);font-size:13px;padding:24px 0;text-align:center}
.foot{margin-top:14px;font-size:11px;color:var(--faint);text-align:center}
</style>
</head>
<body>

<header class="topbar">
  <div class="row1">
    <h1>GUFO · MODEL STATE</h1>
    <div class="pills">
      <span class="pill c-gray" data-p="idle"><i class="dot"></i>Idle</span>
      <span class="pill c-blue" data-p="reading"><i class="dot"></i>Reading</span>
      <span class="pill c-green" data-p="gen"><i class="dot"></i>Generating</span>
      <span class="pill c-amber" data-p="queued"><i class="dot"></i>Queued</span>
      <span class="pill c-red" data-p="error"><i class="dot"></i>Error</span>
    </div>
  </div>
  <div class="row2">
    <span id="stateText">connecting…</span>
    <span id="lastReq"></span>
  </div>
</header>

<main>
<section class="grid">

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/></svg>Model</h2>
    <div class="big" id="mName" style="font-size:22px">—</div>
    <div class="sub" id="mSub">—</div>
    <div style="margin-top:8px">
      <div class="kv"><span>Decode avg 30 min</span><b class="mono-num" id="mDecAvg">—</b></div>
      <div class="kv"><span>Prefill avg 30 min</span><b class="mono-num" id="mPreAvg">—</b></div>
      <div class="kv"><span>Per request (in · out)</span><b class="mono-num" id="mPerReq">—</b></div>
      <div class="kv"><span>Prompt reuse</span><b class="mono-num" id="mReuse">—</b></div>
      <div class="kv"><span>Speculative acceptance (MTP)</span><b class="mono-num" id="mSpec">—</b></div>
      <div class="kv"><span>Requests · since</span><b class="mono-num" id="mCount">—</b></div>
    </div>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>Speed</h2>
    <div class="duo">
      <div>
        <div class="big mono-num g" id="sDec">—<span class="unit">t/s</span></div>
        <div class="sub">Decode · instant</div>
        <div class="sub mono-num" id="sDecRange" style="color:var(--faint)">—</div>
      </div>
      <div>
        <div class="big mono-num t" id="sPre">—<span class="unit">t/s</span></div>
        <div class="sub">Prefill · instant</div>
        <div class="sub mono-num" id="sPreRange" style="color:var(--faint)">—</div>
      </div>
    </div>
    <svg class="spark" id="spDec" preserveAspectRatio="none" viewBox="0 0 200 38"></svg>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h4M7 14h7"/></svg>GPU Load</h2>
    <div class="big mono-num" id="gBusy">—<span class="unit">%</span></div>
    <div class="sub" id="gBusySub">iGPU · sysfs</div>
    <div class="bar"><i id="gBar" style="width:0"></i></div>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><path d="M4 18V8m5 10V4m5 14v-8m5 8V6"/></svg>GPU Memory (GTT)</h2>
    <div class="big mono-num" id="gGtt">—<span class="unit" id="gGttTot"></span></div>
    <div class="sub" id="gGttSub">APU real memory (GTT)</div>
    <div class="bar"><i id="gttBar" style="width:0"></i></div>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><path d="M12 3v10m-3-7a3 3 0 1 1 6 0c0 2-1.5 2.5-1.5 4"/><circle cx="12" cy="17" r="3"/></svg>GPU Temperature</h2>
    <div class="big mono-num a" id="gTemp">—<span class="unit">°C</span></div>
    <div class="sub" id="gTempSub">hwmon</div>
    <div class="bar"><i id="tempBar" style="width:0;background:var(--amber)"></i></div>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>Power</h2>
    <div class="big mono-num" id="gPow">—<span class="unit">W</span></div>
    <div class="sub" id="gPowSub">—</div>
    <div class="bar"><i id="powBar" style="width:0"></i></div>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/></svg>CPU</h2>
    <div class="big mono-num" id="gCpu">—<span class="unit">%</span></div>
    <div class="sub" id="gCpuSub">host · /proc</div>
    <svg class="spark" id="spCpu" preserveAspectRatio="none" viewBox="0 0 200 38"></svg>
  </div>

  <div class="card">
    <h2><svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/></svg>Disk</h2>
    <div class="big mono-num b" id="gDr">—<span class="unit">MB/s</span></div>
    <div class="sub mono-num" id="gDw">—</div>
    <svg class="spark" id="spDisk" preserveAspectRatio="none" viewBox="0 0 200 38"></svg>
  </div>

</section>
)html"
         R"html(
<section class="bottom">

  <div class="panel">
    <h2>Context Fill</h2>
    <div class="donut-wrap">
      <svg width="190" height="190" viewBox="0 0 190 190" id="donut"></svg>
    </div>
    <div class="legend">
      <span><i style="background:var(--green)"></i>In context <b class="mono-num" id="lgUsed" style="margin-left:3px">—</b></span>
      <span><i style="background:var(--gray)"></i>Free <b class="mono-num" id="lgFree" style="margin-left:3px">—</b></span>
      <span><i style="background:var(--faint)"></i>Peak <b class="mono-num" id="lgPeak" style="margin-left:3px">—</b></span>
    </div>
    <div class="hbar"><div class="lbl"><span>GTT (GPU memory)</span><b class="mono-num" id="hbGttV">—</b></div>
      <div class="trk"><i id="hbGtt" style="background:var(--green);width:0"></i></div></div>
    <div class="hbar"><div class="lbl"><span>System RAM</span><b class="mono-num" id="hbRamV">—</b></div>
      <div class="trk"><i id="hbRam" style="background:var(--red);width:0"></i></div></div>
    <div class="hbar"><div class="lbl"><span>GPU Temperature</span><b class="mono-num" id="hbTmpV">—</b></div>
      <div class="trk"><i id="hbTmp" style="background:var(--amber);width:0"></i></div></div>
  </div>

  <div class="panel">
    <h2>Recent Requests <button class="btn" id="showAll">show all</button></h2>
    <div style="overflow-x:auto"><table>
      <thead><tr>
        <th>Time</th><th>Status</th><th>Prompt</th><th>Cached</th><th>Output</th>
        <th>Prefill t/s</th><th>Decode t/s</th><th>TTFT</th><th>Duration</th>
      </tr></thead>
      <tbody id="reqBody"></tbody>
    </table></div>
    <div class="foot" id="reqFoot"></div>
  </div>

</section>
</main>

<div id="sparkTip"></div>
<div class="foot">Gufo (hud branch) · engine-native data · 4 s refresh</div>

<script>
const $=id=>document.getElementById(id);
const nf=(n,d=0)=>n==null?'—':Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const gb=b=>b==null?null:b/1073741824;
const kfmt=n=>n==null?'—':(n>=1000?(n/1000).toLocaleString('en-US',{maximumFractionDigits:1})+'k':String(n));
const sparkTip=$('sparkTip');

let showAllRows=false, ranges={};

function tsStr(ts){
  const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');
  return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
}
function sparkScale(el){
  const pts=el.__pts,W=200,H=38,vals=pts.map(p=>p[1]);
  let lo=Math.min(...vals),hi=Math.max(...vals);
  if(hi-lo<1e-9){hi=lo+1;}
  return {X:i=>i/(pts.length-1)*W, Y:v=>H-2-(v-lo)/(hi-lo)*(H-6)};
}
function placeHover(el,i){
  const pts=el.__pts;if(!pts||pts.length<2)return;
  const {X,Y}=sparkScale(el);
  const x=X(i),y=Y(pts[i][1]);
  const r=el.getBoundingClientRect();
  const sx=Math.max(1e-6,r.width/200); // preserveAspectRatio=none estira en X
  const g=el.querySelector('.hov');
  if(g){
    g.style.display='';
    const ln=g.querySelector('line');
    if(ln){ln.setAttribute('x1',x);ln.setAttribute('x2',x);}
    const c=g.querySelector('ellipse');
    if(c){
      c.setAttribute('cx',x);c.setAttribute('cy',y);
      c.setAttribute('rx',3.4/sx);c.setAttribute('ry',3.4);
    }
  }
  if(!sparkTip)return;
  sparkTip.innerHTML=`<div class="t1">${tsStr(pts[i][0])}</div>`+
    `<div class="t2"><span class="sw" style="background:${el.__color}"></span>`+
    `<span class="lb">${el.__label}</span><span class="vl">${el.__fmt(pts[i][1])}</span></div>`;
  sparkTip.style.display='block';
  const px=r.left+x/200*r.width, py=r.top+y/38*r.height;
  const tw=sparkTip.offsetWidth, th=sparkTip.offsetHeight;
  let left=px+12, top=py+12;
  if(left+tw>window.innerWidth-8){left=px-tw-12;}
  if(top+th>window.innerHeight-8){top=py-th-12;}
  sparkTip.style.left=left+'px';
  sparkTip.style.top=top+'px';
}
function bindSparkHover(el){
  el.__hovBound=true;
  el.addEventListener('mousemove',e=>{
    const pts=el.__pts;if(!pts||pts.length<2)return;
    const r=el.getBoundingClientRect();
    const xf=(e.clientX-r.left)/r.width*200;
    let i=Math.round(xf/(200/(pts.length-1)));
    i=Math.max(0,Math.min(pts.length-1,i));
    el.__hoverI=i;
    placeHover(el,i);
  });
  el.addEventListener('mouseleave',()=>{
    el.__hoverI=null;
    const g=el.querySelector('.hov');
    if(g){g.style.display='none';}
    if(sparkTip){sparkTip.style.display='none';}
  });
}
function spark(el,pts,color,label,fmt){
  if(!el)return;
  const W=200,H=38;
  if(!pts||pts.length<2){el.innerHTML='';el.__pts=null;return;}
  el.__pts=pts;el.__label=label;el.__color=color;
  el.__fmt=fmt||(v=>nf(v,1));
  const {X,Y}=sparkScale(el);
  const line=pts.map((p,i)=>`${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  el.innerHTML=
    `<polyline points="${line}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`+
    `<polygon points="0,${H} ${line} ${W},${H}" fill="${color}" opacity="0.10"/>`+
    `<g class="hov" style="display:none"><line y1="0" y2="${H}" stroke="rgba(255,255,255,.55)" stroke-width="1" stroke-dasharray="3,3" vector-effect="non-scaling-stroke"/>`+
    `<ellipse rx="3.4" ry="3.4" fill="${color}" stroke="rgba(255,255,255,.9)" stroke-width="1.4" vector-effect="non-scaling-stroke"/></g>`;
  if(!el.__hovBound){bindSparkHover(el);}
  if(el.__hoverI!=null){placeHover(el,el.__hoverI);}
}

function donut(pct,used,total){
  const R=76,C=2*Math.PI*R,off=C*(1-Math.max(0,Math.min(1,(pct||0)/100)));
  $('donut').innerHTML=`
    <circle cx="95" cy="95" r="${R}" fill="none" stroke="var(--gray)" stroke-width="14"/>
    <circle cx="95" cy="95" r="${R}" fill="none" stroke="var(--green)" stroke-width="14"
      stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${off}"
      transform="rotate(-90 95 95)" style="transition:stroke-dashoffset .7s"/>
    <text x="95" y="88" text-anchor="middle" fill="var(--txt)" font-size="34" font-weight="700">${nf(pct||0)}%</text>
    <text x="95" y="114" text-anchor="middle" fill="var(--dim)" font-size="14">${kfmt(used)} / ${kfmt(total)}</text>`;
}

const ST_MAP={stop:['Done','var(--green)'],length:['Length','var(--amber)'],abort:['Aborted','var(--red)']};

function render(d){
  const p=d.prom||{},m=d.model||{},reqs=d.requests||[];

  // state
  const reading=p.reading||0, generating=p.generating||0, deferred=p.deferred||0;
  const lost=(p.device_lost||0)>0;
  const st=lost?'error':deferred>0?'queued':generating>0?'gen':reading>0?'reading':'idle';
  document.querySelectorAll('.pill').forEach(el=>el.classList.toggle('on',el.dataset.p===st));
  $('stateText').innerHTML=
    st==='queued'?`<b>${nf(deferred)}</b> request(s) queued`:
    st==='gen'?'<b>Generating…</b>':st==='reading'?'<b>Reading prompt…</b>':
    st==='error'?'<b class="r">device losses detected</b>':'Waiting for a request';
  const last=reqs[0];
  $('lastReq').innerHTML=last?`last: <b>${nf(last.out)} tokens</b> at <b>${nf(last.decode_tps,1)} tok/s</b>`:'';

  // model
  $('mName').textContent=m.served||'—';
  const bits=[];
  if(m.context)bits.push((m.context/1024)+'K context');
  if(m.sessions)bits.push(m.sessions+' sessions');
  if(m.spec)bits.push('spec '+m.spec.toUpperCase());
  $('mSub').textContent=bits.join(' · ')||'—';
  $('mDecAvg').textContent=p.decode_avg!=null?nf(p.decode_avg,1)+' t/s':'—';
  $('mPreAvg').textContent=p.prefill_avg!=null?nf(p.prefill_avg,0)+' t/s':'—';
  const inAvg=reqs.length?reqs.reduce((a,r)=>a+(r.prompt||0),0)/reqs.length:null;
  const outAvg=reqs.length?reqs.reduce((a,r)=>a+(r.out||0),0)/reqs.length:null;
  $('mPerReq').textContent=reqs.length?`${nf(inAvg,0)} in · ${nf(outAvg,0)} out`:'—';
  $('mReuse').textContent=p.prompt_reuse!=null?nf(p.prompt_reuse,0)+' %':'—';
  $('mSpec').textContent=p.spec_accept!=null?nf(p.spec_accept,1)+' %':'—';
  let since='—';
  if(m.started){try{const t=new Date(m.started);since=t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});}catch(e){}}
  $('mCount').textContent=`${nf(reqs.length)} · ${since}`;

  // speed
  $('sDec').innerHTML=nf(p.decode_now,1)+'<span class="unit">t/s</span>';
  $('sPre').innerHTML=nf(p.prefill_now,0)+'<span class="unit">t/s</span>';
  $('sDecRange').textContent=p.decode_min!=null?`30 min: ${nf(p.decode_min,0)} – ${nf(p.decode_max,0)} t/s`:'';
  $('sPreRange').textContent=p.prefill_min!=null?`30 min: ${nf(p.prefill_min,0)} – ${nf(p.prefill_max,0)} t/s`:'';

  // gpu
  $('gBusy').innerHTML=nf(p.gpu_busy)+'<span class="unit">%</span>';
  $('gBar').style.width=(p.gpu_busy||0)+'%';
  $('gBar').style.background=p.gpu_busy>90?'var(--amber)':'var(--green)';

  const gu=gb(p.gtt_used),gt=gb(p.gtt_total);
  $('gGtt').innerHTML=(gu==null?'—':nf(gu,1))+`<span class="unit">${gt?'/ '+nf(gt,0)+' GB':''}</span>`;
  $('gttBar').style.width=p.gtt_total?((p.gtt_used/p.gtt_total)*100)+'%':'0';

  $('gTemp').innerHTML=nf(p.gpu_temp)+'<span class="unit">°C</span>';
  $('tempBar').style.width=p.gpu_temp?Math.min(100,p.gpu_temp)+'%':'0';
  $('gTempSub').textContent=p.gpu_temp>85?'hwmon · running hot':'hwmon';

  $('gPow').innerHTML=nf(p.power_w,1)+'<span class="unit">W</span>';
  $('gPowSub').textContent=p.power_avg_w!=null?`avg ${nf(p.power_avg_w,1)} W · hwmon`:'hwmon';
  $('powBar').style.width=p.power_w?Math.min(100,p.power_w/120*100)+'%':'0';

  $('gCpu').innerHTML=nf(p.cpu_pct,0)+'<span class="unit">%</span>';

  $('gDr').innerHTML=(p.disk_read==null?'—':nf(p.disk_read/1048576,1))+'<span class="unit">MB/s</span>';
  $('gDw').textContent=p.disk_write==null?'':`write ${nf(p.disk_write/1048576,1)} MB/s`;

  // context
  const ctx=m.context||p.n_tokens_max||0;
  const used=p.kv_pct!=null&&ctx?Math.round(p.kv_pct/100*ctx):null;
  donut(p.kv_pct,used,ctx);
  $('lgUsed').textContent=kfmt(used);
  $('lgFree').textContent=ctx?kfmt(ctx-(used||0)):'—';
  $('lgPeak').textContent=kfmt(p.n_tokens_max);
  $('hbGttV').textContent=gu!=null?`${nf(gu,1)} / ${nf(gt,0)} GB`:'—';
  $('hbGtt').style.width=p.gtt_total?((p.gtt_used/p.gtt_total)*100)+'%':'0';
  const ru=gb(p.ram_used),rt=gb(p.ram_total),rp=rt?ru/rt*100:0;
  $('hbRamV').textContent=ru!=null?`${nf(ru,1)} / ${nf(rt,1)} GB`:'—';
  $('hbRam').style.width=rp+'%';
  $('hbTmpV').textContent=p.gpu_temp!=null?nf(p.gpu_temp)+' °C':'—';
  $('hbTmp').style.width=p.gpu_temp?Math.min(100,p.gpu_temp)+'%':'0';

  // table
  const rows=(showAllRows?reqs:reqs.slice(0,12));
  $('showAll').textContent=showAllRows?`close (${reqs.length})`:`show all (${reqs.length})`;
  $('reqFoot').textContent=reqSummary(reqs);
  $('reqBody').innerHTML=rows.map(r=>{
    const [lbl,col]=ST_MAP[r.status]||[r.status,'var(--dim)'];
    const t=r.ts?new Date(r.ts*1000).toLocaleTimeString('en-US'):'—';
    return `<tr><td class="mono-num">${t}</td>
      <td><span class="st" style="color:${col}"><i style="background:${col}"></i>${lbl}</span></td>
      <td class="mono-num">${nf(r.prompt)}</td>
      <td class="mono-num">${r.cached!=null?'<span class="g">'+nf(r.cached)+'</span>':'—'}</td>
      <td class="mono-num">${nf(r.out)}</td>
      <td class="mono-num">${nf(r.prefill_tps,0)}</td>
      <td class="mono-num">${nf(r.decode_tps,1)}</td>
      <td class="mono-num">${r.ttft_ms!=null?nf(r.ttft_ms,0)+' ms':'—'}</td>
      <td class="mono-num">${r.dur!=null?nf(r.dur/1000,1)+' s':'—'}</td></tr>`;
  }).join('')||`<tr><td colspan="9" class="empty">no requests yet</td></tr>`;
}

// The sentence under the request table, in plain words (Strata's Monitor
// style): how many requests, average output and prompt-reading speed, and
// how much prompt was reused. Parts with no sample are left out.
function reqSummary(reqs){
  if(!reqs.length)return 'No requests yet';
  const d=reqs.map(r=>r.decode_tps||0).filter(v=>v>0);
  const pf=reqs.map(r=>r.prefill_tps||0).filter(v=>v>0);
  const out=d.length?d.reduce((a,b)=>a+b,0)/d.length:null;
  const reading=pf.length?pf.reduce((a,b)=>a+b,0)/pf.length:null;
  const reused=reqs.reduce((a,r)=>a+(r.cached||0),0);
  let s='In '+nf(reqs.length)+' requests';
  if(out!=null)s+=' your average output was '+nf(out,1)+' tok/s';
  if(reading!=null)s+=(out!=null?' and prompt reading ':' reading ')+nf(reading,0)+' tok/s';
  if(reused>0)s+=', reusing '+kfmt(reused)+' tokens';
  return s+'.';
}

function renderRanges(){
  spark($('spDec'),ranges.decode,'var(--green)','decode',v=>nf(v,1)+' tok/s');
  spark($('spCpu'),ranges.cpu_pct,'var(--green)','cpu',v=>nf(v,0)+'%');
  spark($('spDisk'),ranges.disk_read,'var(--blue)','disk read',v=>nf(v/1048576,1)+' MB/s');
}

async function poll(){
  try{
    const r=await fetch('/hud/api/state');
    const d=await r.json();
    if(d.ranges){ranges=d.ranges;renderRanges();}
    render(d);
    document.title='Gufo · '+(d.prom&&d.prom.decode_now!=null?nf(d.prom.decode_now,0)+' t/s':'idle');
  }catch(e){$('stateText').textContent='HUD has no data: '+e.message;}
}
$('showAll').onclick=()=>{showAllRows=!showAllRows;poll();};
poll();
setInterval(poll,4000);
</script>
</body>
</html>)html";
}

}  // namespace gufo::server
