import cv2, glob, os, numpy as np, collections
V='/home/ubuntu/tryan/apd/data/dump/dataset/tc_2272_v11s_640/data/val'
out=collections.defaultdict(list)
for lf in glob.glob(V+'/labels/*.txt'):
    im=cv2.imread(V+'/images/'+os.path.basename(lf)[:-4]+'.jpg')
    if im is None: continue
    H,W=im.shape[:2]
    for ln in open(lf):
        p=ln.split()
        c=int(p[0]); v=list(map(float,p[1:])); xs=v[0::2]; ys=v[1::2]
        x1,y1,x2,y2=int(min(xs)*W),int(min(ys)*H),int(max(xs)*W),int(max(ys)*H)
        if x1<0 or y1<0 or x2>W or y2>H: continue
        cr=im[y1:y2,x1:x2]
        ar=(x2-x1)/max(1,(y2-y1))
        if 1.1<ar<2.2 and (x2-x1)>=120: out[c].append((-(x2-x1)*(y2-y1),lf,cr))
os.makedirs('sprites',exist_ok=True)
rows=[]
for c in range(6):
    L=sorted(out[c],key=lambda t:t[0])[:8]
    print(c,len(out[c]))
    tiles=[]
    for i,(_,lf,cr) in enumerate(L):
        cv2.imwrite(f'sprites/c{c}_{i}.png',cr)
        t=cv2.resize(cr,(160,100)); cv2.putText(t,f'{c}_{i}',(3,14),0,0.5,(0,0,255),2); tiles.append(t)
    while len(tiles)<8: tiles.append(np.zeros((100,160,3),np.uint8))
    rows.append(np.hstack(tiles))
cv2.imwrite('sheet.jpg',np.vstack(rows))
