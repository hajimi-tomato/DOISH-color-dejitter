const q = s => element.querySelector(s);
const viewport = q('.viewport'), canvas = q('canvas'), ctx = canvas.getContext('2d');
let images = [], displayed = [], scale = 1, dx = 0, dy = 0, drag = null, version = 0, frame = 0;
function fit() { scale = 1; dx = dy = 0; draw(); }
function draw() {
  const w = viewport.clientWidth, h = viewport.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  q('.zoom-label').textContent = `${Math.round(scale*100)}%`;
  if (!displayed.length) return;
  const [a,b] = displayed, fitScale = Math.min(w/a.width,h/a.height);
  const iw = a.width*fitScale*scale, ih = a.height*fitScale*scale, x = (w-iw)/2+dx, y = (h-ih)/2+dy;
  const split = w * Number(q('.split').value)/100;
  ctx.save(); ctx.beginPath(); ctx.rect(0,0,split,h); ctx.clip(); ctx.drawImage(a,x,y,iw,ih); ctx.restore();
  ctx.save(); ctx.beginPath(); ctx.rect(split,0,w-split,h); ctx.clip(); ctx.drawImage(b,x,y,iw,ih); ctx.restore();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(split,0);ctx.lineTo(split,h);ctx.stroke();
}
function adjusted(image) {
  const hShift = Number(q('.hue').value), sGain = Number(q('.saturation').value)/100, vGain = Number(q('.brightness').value)/100;
  if (hShift===0 && sGain===1 && vGain===1) return image;
  const c = document.createElement('canvas'); c.width=image.width; c.height=image.height;
  const g=c.getContext('2d',{willReadFrequently:true}); g.drawImage(image,0,0);
  const pixels=g.getImageData(0,0,c.width,c.height), p=pixels.data;
  for(let i=0;i<p.length;i+=4){
    const r=p[i]/255, g0=p[i+1]/255, b=p[i+2]/255, max=Math.max(r,g0,b), min=Math.min(r,g0,b), d=max-min;
    let h=0; if(d) h=max===r ? ((g0-b)/d)%6 : max===g0 ? (b-r)/d+2 : (r-g0)/d+4;
    h=((h*60+hShift)%360+360)%360/60;
    const s=Math.min(1,(max ? d/max : 0)*sGain), v=Math.min(1,max*vGain), chroma=v*s, x=chroma*(1-Math.abs(h%2-1)), m=v-chroma;
    const values=h<1?[chroma,x,0]:h<2?[x,chroma,0]:h<3?[0,chroma,x]:h<4?[0,x,chroma]:h<5?[x,0,chroma]:[chroma,0,x];
    p[i]=(values[0]+m)*255; p[i+1]=(values[1]+m)*255; p[i+2]=(values[2]+m)*255;
  }
  g.putImageData(pixels,0,0); return c;
}
function preview() {
  q('.h-val').textContent=q('.hue').value+'°'; q('.s-val').textContent=q('.saturation').value+'%';q('.b-val').textContent=q('.brightness').value+'%';
  cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{displayed=images.map(adjusted);draw();});
}
async function load() {
  const current=++version;
  let value; try{value=JSON.parse(props.value||'{}');}catch{value={};}
  if(!value.original){images=displayed=[];q('.empty').style.display='grid';fit();return;}
  const urls=[value.original,value.result||value.original];
  try {
    const loaded=await Promise.all(urls.map(url=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=url;})));
    if(current!==version)return;
    images=loaded;q('.empty').style.display='none';q('.result').textContent=value.result?'结果':'待处理';
    scale=1;dx=dy=0;preview();
  } catch { if(current===version){images=displayed=[];q('.empty').textContent='预览加载失败，请重新上传图片';q('.empty').style.display='grid';draw();} }
}
viewport.addEventListener('wheel',e=>{
  if(!images.length)return;e.preventDefault();
  const old=scale;scale=Math.max(0.5,Math.min(16,scale*Math.exp(-e.deltaY*0.0015)));
  const r=viewport.getBoundingClientRect(), px=e.clientX-r.left-r.width/2, py=e.clientY-r.top-r.height/2;
  dx=px-(px-dx)*scale/old;dy=py-(py-dy)*scale/old;draw();
},{passive:false});
viewport.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,dx,dy};viewport.setPointerCapture(e.pointerId);});
viewport.addEventListener('pointermove',e=>{if(drag){dx=drag.dx+e.clientX-drag.x;dy=drag.dy+e.clientY-drag.y;draw();}});
for(const event of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(event,()=>{drag=null;});
viewport.addEventListener('dblclick',fit);
viewport.addEventListener('keydown',e=>{if(e.key==='0'){fit();e.preventDefault();}});
q('.reset-view').addEventListener('click',fit);q('.split').addEventListener('input',draw);
for(const name of ['.hue','.saturation','.brightness'])q(name).addEventListener('input',preview);
q('.reset-hsb').addEventListener('click',()=>{q('.hue').value=0;q('.saturation').value=100;q('.brightness').value=100;preview();});
const observer=new ResizeObserver(draw);observer.observe(viewport);
watch('value',load);load();
