"""Write a candidate CSV: python3 mk.py NAME "G1 group" "G2 group" "G3 group"  (two-digit endpoint ids, space separated)."""
import sys
FAM={'G1':"01 02 04 12 14 17 18 23 24 25 33",'G2':"03 06 07 08 20 21 22 26 28 30 32",'G3':"05 09 10 11 13 15 16 19 27 29 31"}
name=sys.argv[1]; groups=dict(zip(['G1','G2','G3'],sys.argv[2:5])); lab={}
for g,m in FAM.items():
    members=groups[g].split(); assert all(x in m.split() for x in members),(g,members)
    for x in m.split(): lab[x]=(g,f'{g}-grp0' if x in members else f'{g}-{x}')
open(f'candidates/{name}.csv','w').write("endpoint,family,model\n"+"\n".join(f"ep-{x},{lab[x][0]},{lab[x][1]}" for x in sorted(lab))+"\n")
print(name,{g:v for g,v in groups.items()})
