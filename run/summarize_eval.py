#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize LMMS-Eval results under a base directory.

Scans subfolders for metrics.json or aggregated CSVs and prints a quick table.
"""
import os, json, sys

def find_metrics(root: str):
	out = []
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if fn.endswith('metrics.json'):
				path = os.path.join(dirpath, fn)
				try:
					with open(path, 'r') as f:
						data = json.load(f)
				except Exception:
					continue
				# heuristic: extract a few key metrics if present
				rec = {'path': path}
				for key in ['acc', 'accuracy', 'overall', 'avg', 'score']:
					if key in data and isinstance(data[key], (int, float)):
						rec[key] = data[key]
				# task name hint
				parts = dirpath.split(os.sep)
				for p in parts[::-1]:
					if p and p.lower() not in ('results','logs','lmms_eval'):
						rec['task'] = p
						break
				out.append(rec)
	return out

def main():
	base = sys.argv[1] if len(sys.argv) > 1 else './eval_runs'
	metrics = find_metrics(base)
	if not metrics:
		print(f"No metrics.json found under {base}")
		return
	# group by run tag (top-level folder under base)
	runs = {}
	for rec in metrics:
		# find run tag as the folder directly under base
		rel = os.path.relpath(rec['path'], base)
		parts = rel.split(os.sep)
		run = parts[0]
		runs.setdefault(run, []).append(rec)
	# print summary
	for run, recs in sorted(runs.items()):
		print(f"\n=== {run} ===")
		for r in sorted(recs, key=lambda x: x.get('task','')):
			score = r.get('acc') or r.get('accuracy') or r.get('overall') or r.get('avg') or r.get('score')
			print(f"  - {r.get('task','?')}: {score}  ({r['path']})")

if __name__ == '__main__':
	main()

