/* All arrays are captured model/log data. The page performs no inference. */
(() => {
  'use strict';
  const data=window.CHICGRASP_DATA;
  const $=id=>document.getElementById(id);
  if(!data){$('actionReadout').textContent='The trace data could not load. Download the action-generation video below.';return;}
  const color={teal:'#1f77b4',gold:'#ff7f0e',blue:'#2ca02c',dim:'#555555',white:'#444444',grid:'#cccccc'};
  let animation=null;
  function setup(id){const c=$(id),r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);return {ctx,w:r.width,h:r.height};}
  function line(ctx,pts,stroke,width=2){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();}
  function label(ctx,value,x,y,size=11,fill=color.dim){ctx.font=`${size}px system-ui, sans-serif`;ctx.fillStyle=fill;ctx.fillText(value,x,y);}
  function render(){
    const si=Number($('sample').value),step=Number($('step').value),sample=data.denoising.samples[si],physical=sample.physical[step];
    let v=sample.normalized[step].map(x=>x.slice(0,3));
    const zoom=$('zoom').checked;
    let span=0;
    if(zoom){const lo=[0,1,2].map(j=>Math.min(...v.map(p=>p[j]))),hi=[0,1,2].map(j=>Math.max(...v.map(p=>p[j])));span=Math.max(...hi.map((x,j)=>x-lo[j]));v=v.map(p=>p.map((x,j)=>(x-(hi[j]+lo[j])/2)/Math.max(span,1e-5)*3));}
    const {ctx,w,h}=setup('trajectory');
    const project=p=>[w*.5+(p[0]*.78-p[1]*.62)*h*.18,h*.55+(p[0]*.3+p[1]*.4-p[2]*.9)*h*.18];
    const origin=project([0,0,0]);
    [[2.5,0,0],[0,2.5,0],[0,0,2.5]].forEach((p,i)=>{const end=project(p);line(ctx,[origin,end],color.grid,1);label(ctx,['x','y','z'][i],end[0]+5,end[1],[12][0],[color.teal,color.gold,color.blue][i]);});
    const uv=v.map(project);line(ctx,uv,color.teal,2.5);uv.forEach((p,i)=>{ctx.beginPath();ctx.arc(...p,i===0?4:2.6,0,Math.PI*2);ctx.fillStyle=i>=1&&i<=6?color.teal:color.white;ctx.fill();});
    label(ctx,zoom?'Zoomed to current plan · equal XYZ scale':'Fixed scale throughout denoising',15,h-17,10);
    if(step===100&&!zoom)label(ctx,'Enable “Zoom to plan” to inspect the final targets.',15,23,10,color.teal);
    if(zoom)label(ctx,`Displayed extent: ${span.toPrecision(3)} normalized units`,15,23,10,color.teal);
    const j=setup('jaws'),x0=42,y0=30,pw=j.w-68,ph=j.h-72;
    // Fixed range preserves visibility of values outside the binary command domain.
    const y=value=>y0+(2-value)/3*ph;
    [-1,0,.5,1,2].forEach(value=>{j.ctx.setLineDash(value===.5?[4,4]:[]);line(j.ctx,[[x0,y(value)],[x0+pw,y(value)]],value===.5?color.dim:color.grid,1);label(j.ctx,String(value),10,y(value)+4,10);});j.ctx.setLineDash([]);
    [6,7].forEach((dim,i)=>{const pts=physical.map((p,k)=>[x0+k/15*pw,y(p[dim])]);line(j.ctx,pts,[color.teal,color.gold][i],2.5);});
    [1,6,16].forEach(index=>label(j.ctx,String(index),x0+(index-1)/15*pw-4,j.h-24,10));label(j.ctx,'Planned action index',x0+pw*.3,j.h-6,10);
    $('iteration').innerHTML=`${step} <small>/ 100</small>`;$('stepReadout').textContent=`${step} / 100`;
    if(step===100){const p=physical[1],binary=x=>Math.max(0,Math.min(1,Math.round(x))),L=binary(p[6]),R=binary(p[7]);$('actionReadout').textContent=`First future target: x ${p[0].toFixed(3)} m · y ${p[1].toFixed(3)} m · z ${p[2].toFixed(3)} m | Left ${L?'OPEN':'CLOSED'} · Right ${R?'OPEN':'CLOSED'}. Trained action prefix: indices 2–7 of the 16-step window.`;}
    else $('actionReadout').textContent='Denoising in progress. Intermediate noisy values are not actuator commands.';
  }
  function stop(){if(animation!==null)cancelAnimationFrame(animation);animation=null;$('play').textContent='▶ Replay denoising';}
  $('play').addEventListener('click',()=>{if(animation!==null){stop();return;}$('step').value=0;$('play').textContent='Ⅱ Pause';const start=performance.now();const tick=now=>{$('step').value=Math.min(100,Math.floor((now-start)/60));render();if(Number($('step').value)<100)animation=requestAnimationFrame(tick);else stop();};animation=requestAnimationFrame(tick);});
  $('step').addEventListener('input',()=>{stop();render();});$('zoom').addEventListener('change',render);
  $('sample').addEventListener('change',()=>{stop();const time=data.denoising.samples[Number($('sample').value)].time;for(let c=0;c<3;c++)$('obs'+c).src=`assets/data/t${time}_camera_${c}.jpg`;render();});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  const rec=data.recorded,video=$('recordedVideo');
  function recorded(){const t=video.currentTime;let i=0;while(i<rec.time.length-1&&rec.time[i+1]<=t)i++;const initialized=rec.action[i].some(x=>x!==0);$('leftState').textContent=initialized?(rec.action[i][6]>=.5?'OPEN':'CLOSED'):'Initializing';$('rightState').textContent=initialized?(rec.action[i][7]>=.5?'OPEN':'CLOSED'):'Initializing';$('leftState').style.color='var(--teal)';$('rightState').style.color='var(--gold)';$('recordedTime').textContent=t.toFixed(1)+' s';}
  ['timeupdate','seeked','loadedmetadata'].forEach(event=>video.addEventListener(event,recorded));
  $('jump').addEventListener('click',()=>{const start=()=>{video.currentTime=18;video.play().catch(()=>{});};if(video.readyState>=1)start();else{video.addEventListener('loadedmetadata',start,{once:true});video.load();}});
  render();recorded();
})();
