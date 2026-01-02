#!/usr/bin/env python3
import os, sys, json, glob
from math import isfinite

def group_rows(lines, y_tol=6.0):
    rows=[]; cur=[]
    last_y=None
    for ln in sorted(lines, key=lambda x: (x["bbox"][1], x["bbox"][0])):
        y = (ln["bbox"][1]+ln["bbox"][3])/2.0
        if last_y is None or abs(y-last_y)<=y_tol:
            cur.append(ln); last_y = y if last_y is None else (last_y+y)/2
        else:
            if cur: rows.append(cur)
            cur=[ln]; last_y=y
    if cur: rows.append(cur)
    return rows

def split_cells(line_text):
    # simple heuristic: split on 2+ spaces; improve if needed
    import re
    parts = [p.strip() for p in re.split(r'\s{2,}', line_text.strip()) if p.strip()]
    return parts if parts else [line_text.strip()]

def build_tables(tags):
    tables=[]
    # gather table-row candidates
    tr = [t for t in tags if t["tag"]=="TableRowCandidate"]
    if not tr: return tables
    # optional: only inside explicit table regions
    rows = group_rows(tr, y_tol= max(6.0, sum(t["size"] for t in tr)/max(1,len(tr))/2))
    for ridx, row in enumerate(rows, start=1):
        cells = [split_cells(t["text"]) for t in row]
        # flatten row-level by merging lines per row into one string before split
        merged = " ".join(t["text"] for t in row)
        split = split_cells(merged)
        tables.append({"row": ridx, "cells": split})
    return tables

def build_lists(tags):
    lists=[]; cur=[]
    for t in tags:
        if t["tag"]=="LI":
            cur.append(t)
        else:
            if cur: lists.append(cur); cur=[]
    if cur: lists.append(cur)
    # convert to serializable blocks
    out=[]
    for i, block in enumerate(lists, start=1):
        out.append({
            "list_index": i,
            "type": "OL" if any(x["text"].strip().split(" ",1)[0].rstrip(".").isdigit() for x in block) else "UL",
            "items": [x["text"] for x in block]
        })
    return out

def main(fused_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for tp in sorted(glob.glob(os.path.join(fused_dir, "page_*.fused.json"))):
        with open(tp,"r",encoding="utf-8") as f: T=json.load(f)
        pnum = T["page"]; tags = T["tags"]
        tables = build_tables(tags)
        lists  = build_lists(tags)
        with open(os.path.join(out_dir, f"page_{pnum:03d}.tables.json"),"w",encoding="utf-8") as f:
            json.dump({"page":pnum,"tables":tables}, f, indent=2)
        with open(os.path.join(out_dir, f"page_{pnum:03d}.lists.json"),"w",encoding="utf-8") as f:
            json.dump({"page":pnum,"lists":lists}, f, indent=2)
        print("wrote tables/lists for page", pnum)

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python 05_tables_lists.py fused_tags structures"); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
