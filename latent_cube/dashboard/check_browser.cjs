const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs'),zlib=require('zlib'),assert=require('assert/strict');
const site=process.argv[2], manifest=JSON.parse(fs.readFileSync(site+'/manifest.json','utf8'));
const initial=manifest.rows.find(r=>r.id==='tay|sin x|2|hold'&&r.variant==='forced');
const originalReply=async locator=>locator.evaluate(el=>[...el.childNodes].map(n=>n.nodeType===1&&n.classList.contains('math-fragment')?n.dataset.original:n.textContent).join(''));
const read=r=>JSON.parse(zlib.gunzipSync(fs.readFileSync(site+'/'+r.path)));
const initialData=read(initial);
(async()=>{
 const browser=await chromium.launch({headless:true});
 const report={layouts:[],interactions:[],errors:[],externalRequests:[],allowedCdnRequests:[]};
 for(const width of [1440,390,320])for(const theme of ['light','dark']){
  const page=await browser.newPage({viewport:{width,height:900},colorScheme:theme});
  page.on('pageerror',e=>report.errors.push(e.message));
  page.on('request',r=>{if(r.url()==='https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js')report.allowedCdnRequests.push(r.url());else if(!r.url().startsWith('http://127.0.0.1:18765/'))report.externalRequests.push(r.url())});
  await page.goto('http://127.0.0.1:18765/');
  await page.waitForFunction(()=>document.querySelectorAll('#pcaPlot circle').length===338&&document.querySelectorAll('#comparison tbody tr').length===32,{},{timeout:60000});
  assert.equal(await page.evaluate(()=>document.characterSet),'UTF-8');
  assert.equal(await page.locator('#lens h2').textContent(),'If the model stopped here…');
  assert.equal(await page.locator('#promptSelect').inputValue(),String(initial.index));
  assert.equal(await page.locator('#tokenRows tr').count(),initialData.meta.tokens.length);
  assert.equal(await page.locator('#promptInfo .card').nth(2).locator('math').count(),3);
  assert.equal(await page.locator('#normMap tbody tr').count(),32);
  assert.equal(await page.locator('#normMap tbody button').count(),32*initialData.meta.tokens.length);
  assert.equal(await page.locator('#lensMap tbody button').count(),32*initialData.meta.tokens.length);
  assert.equal(await page.locator('#topNeurons tbody tr').count(),20);
  assert.equal(await page.locator('#neuronAcross tbody tr').count(),338);
  assert.equal(await page.locator('#attnMap tbody button').count(),52*52);
  assert.equal(await page.locator('#promptInfo .card').nth(0).locator('.text').textContent(),initialData.meta.prompt);
  assert.equal(await originalReply(page.locator('#promptInfo .card').nth(2).locator('.text')),initialData.meta.free_reply);
  assert.equal(await originalReply(page.locator('#promptInfo .card').nth(3).locator('.text')),initialData.meta.captured_reply);
  const layout=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,background:getComputedStyle(document.body).backgroundColor,mathElements:document.querySelectorAll('math').length}));
  assert.equal(layout.width,layout.scrollWidth); report.layouts.push({...layout,theme});
  if(width===1440&&theme==='light'){
   const replyFields=manifest.rows.flatMap(r=>{const m=read(r).meta;return ['free_reply','captured_reply'].map(key=>({id:m.id,key,template:m.template,text:m[key]}))});
   report.replyPreservation=await page.evaluate(fields=>{
    let expressions=0;
    for(const field of fields){
     const box=document.createElement('div');box.innerHTML=renderReply(field.text,field.template);
     const reconstructed=[...box.childNodes].map(n=>n.nodeType===1&&n.classList.contains('math-fragment')?n.dataset.original:n.textContent).join('');
     if(reconstructed!==field.text)throw Error('Reply content changed: '+field.id+' '+field.key);
     expressions+=box.querySelectorAll('.math-fragment').length;
    }
    return {fields:fields.length,mathExpressions:expressions,failures:0};
   },replyFields);
   await page.selectOption('#lensKind','0');
   await page.locator('#lensMap button').first().focus();
   assert((await page.locator('#lensDetail').textContent()).includes('probability'));
   await page.selectOption('#normKind','R2');await page.locator('#normMap button').first().focus();
   assert((await page.locator('#normDetail').textContent()).includes('R2 mixer addition'));
   await page.selectOption('#attnLayer',{value:'7'});await page.selectOption('#attnHead','15');
   await page.locator('#attnMap button').first().focus();assert((await page.locator('#attnDetail').textContent()).includes('Layer 31, head 15'),await page.locator('#attnDetail').textContent());
   await page.selectOption('#neuronId','9072');await page.selectOption('#neuronReadout','reply_end');
   await page.selectOption('#pcaColour','func');await page.selectOption('#pcaLayer','0');
   await page.waitForFunction(()=>document.querySelector('#pcaLegend').textContent.includes('sin x'));
   await page.locator('#pcaPlot circle').first().focus();assert((await page.locator('#pcaDetail').textContent()).includes(manifest.rows[0].prompt));
   await page.selectOption('#compareB',String(initial.index));
   await page.waitForFunction(()=>[...document.querySelectorAll('#comparison tbody tr')].every(r=>Number(r.children[1].textContent)===1));
   for(const [id,value] of [['func','sin x'],['kth','2'],['template','hold'],['variant','forced'],['power','3']]){
    await page.selectOption('#filter-'+id,value);
    await page.waitForFunction(()=>document.querySelector('#loading').textContent.startsWith('Loaded'));
   }
   assert.equal(await page.locator('#promptSelect option').count(),1);
   report.interactions.push('rank/top-five, norm cell, attention layer/head, neuron/readout, PCA colour/layer/hover, same-capture comparison, all five filters');
   await page.evaluate(()=>{for(const id of ['func','kth','template','variant','power'])document.getElementById('filter-'+id).value='';document.getElementById('filter-func').dispatchEvent(new Event('change'))});
   await page.waitForFunction(()=>document.querySelectorAll('#promptSelect option').length===338);
   const longest=manifest.rows.reduce((a,b)=>a.tokens>b.tokens?a:b),data=read(longest);
   await page.selectOption('#promptSelect',String(longest.index));
   await page.waitForFunction(n=>document.querySelectorAll('#tokenRows tr').length===n,longest.tokens,{timeout:60000});
   assert.equal(await originalReply(page.locator('#promptInfo .card').nth(2).locator('.text')),data.meta.free_reply);
   assert.equal(await originalReply(page.locator('#promptInfo .card').nth(3).locator('.text')),data.meta.captured_reply);
   assert.equal(await page.locator('#normMap td.empty').count(),32*(longest.tokens-data.norm_positions.length));
   assert.equal(await page.locator('#lensMap td.empty').count(),32*(longest.tokens-data.lens_positions.length));
   assert.equal(await page.locator('#attnMap tbody tr').count(),data.attention_queries.length);
   await page.locator('#normMap tbody tr').nth(5).locator('button').first().focus();
   assert.equal(await page.locator('#neuronLayer').inputValue(),'5');
   const neuronPosition=Number(await page.locator('#neuronToken').inputValue());
   const expectedNeuron=data.mlp[5].find(c=>c.position===neuronPosition).top[0];
   assert.equal(Number(await page.locator('#topNeurons tbody tr').first().locator('td').nth(0).textContent()),expectedNeuron[0]);
   assert.equal(Number(await page.locator('#topNeurons tbody tr').first().locator('td').nth(1).textContent()),expectedNeuron[1]);
   for(const w of [1440,390,320]){await page.setViewportSize({width:w,height:900});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),w)}
   report.interactions.push(`longest capture ${longest.id} ${longest.variant}: ${longest.tokens} full tokens, full reply text equality, exact unavailable-cell counts, no page overflow at all three widths`);
   await page.setViewportSize({width:1440,height:900});await page.selectOption('#promptSelect',String(initial.index));
   await page.waitForFunction(()=>document.querySelectorAll('#tokenRows tr').length===52);
  }
  if((width===1440&&theme==='light')||(width===390&&theme==='dark')){
   await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:site+'/../browser-'+width+'-'+theme+'.png',fullPage:true});
   await page.locator('#browser').screenshot({path:site+'/../browser-prompt-'+width+'-'+theme+'.png'});
   await page.locator('#lens').screenshot({path:site+'/../browser-lens-'+width+'-'+theme+'.png'});
  }
  // Test explicit override of system colour scheme.
  await page.selectOption('#theme',theme==='dark'?'light':'dark');
  const overridden=await page.evaluate(()=>getComputedStyle(document.body).backgroundColor);
  assert.notEqual(overridden,layout.background);
  await page.close();
 }
 assert.equal(report.errors.length,0);assert.equal(report.externalRequests.length,0);
 fs.writeFileSync(site+'/../browser_report.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
