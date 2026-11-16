// Optional presentation export for environments providing @oai/artifact-tool.
// The portable Python pipeline also provides a standard openpyxl exporter.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const folder = path.resolve(process.argv[2] || 'outputs/smoke');
const records = Object.values(JSON.parse(await fs.readFile(path.join(folder, 'performance_benchmarks.json'), 'utf8')));
const provenance = JSON.parse(await fs.readFile(path.join(folder, 'results/environment.json'), 'utf8'));
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Benchmarks');
sheet.showGridLines = false;
const fields = ['model_name','test_accuracy','model_size_mb','compression_ratio','accuracy_loss_percent',
  'single_inference_ms','single_p95_ms','batch_inference_ms','batch_size','throughput_images_per_second',
  'peak_memory_mb','flops','flops_reduction','energy_consumption','ece'];
const headers = ['Model','Accuracy','Artifact size (MiB)','Compression (x)','Accuracy loss (pp)',
  'Single latency (ms)','Single p95 (ms)','Batch latency (ms)','Batch size','Throughput (images/s)',
  'Process RSS (MiB)','Estimated FLOPs','FLOPs reduction','Energy estimate (J/image)','ECE'];
sheet.getRange('A2').values = [['Compression performance benchmarks']];
sheet.getRange('A3').values = [[provenance.data_source]];
sheet.getRange('A4').values = [['Source: performance_benchmarks.json from this run. Blank energy values mean no power measurement.']];
sheet.getRange('A6:O6').values = [headers];
sheet.getRange(`A7:O${records.length+6}`).values = records.map(record => fields.map(field=>record[field]??null));
sheet.getRange(`A2:O${records.length+6}`).format.font = {name:'Arial',size:10};
sheet.getRange('A2').format.font = {name:'Arial',size:15,bold:true,color:'#173C50'};
sheet.getRange('A3').format.font = {name:'Arial',size:10,italic:true,color:'#6C4D13'};
sheet.getRange('A6:O6').format = {fill:'#173C50',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center'};
sheet.getRange('A6:O6').format.rowHeight = 38;
sheet.getRange(`A7:O${records.length+6}`).format.rowHeight = 40;
sheet.getRange(`A7:O${records.length+6}`).format.verticalAlignment = 'center';
sheet.getRange(`A7:A${records.length+6}`).format.wrapText = true;
sheet.getRange('A:A').format.columnWidth = 62;
sheet.getRange('B:O').format.columnWidth = 19;
sheet.getRange(`B7:O${records.length+6}`).setNumberFormat('0.000');
for (const column of ['B','M','O']) sheet.getRange(`${column}7:${column}${records.length+6}`).setNumberFormat('0.00%');
for (const column of ['I','L']) sheet.getRange(`${column}7:${column}${records.length+6}`).setNumberFormat('#,##0');
sheet.freezePanes.freezeRows(6);
sheet.freezePanes.freezeColumns(1);
sheet.tables.add(`A6:O${records.length+6}`,true,'PerformanceBenchmarks');
workbook.recalculate();
console.log((await workbook.inspect({kind:'table',range:'Benchmarks!A6:F10',include:'values',tableMaxRows:5,tableMaxCols:6,maxChars:2500})).ndjson);
console.log((await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
for (const [name,range] of [['benchmark_preview','A1:F15'],['benchmark_metrics_preview','G6:O15']]) {
  const preview = await workbook.render({sheetName:'Benchmarks',range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(folder,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(folder,'performance_benchmarks.xlsx'));
