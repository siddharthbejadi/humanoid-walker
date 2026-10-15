"""Teacher labelling: frames -> MoveNet -> plausibility gate -> IK -> robot joint angles [shoulder, arm, elbow] (reachability built in).
    label_arm.py OUT.npz  name=FRAMES_DIR ...  [neg:name=FRAMES_DIR ...]
'neg:' clips (arm not in view, e.g. hand close-ups) contribute only 'present = 0' examples (every 4th frame).
Images are padded to a square (not stretched, same as MoveNet) and resized to 192.  The cache holds camera pixels: PRIVATE."""
import sys, glob, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.pose import MoveNet, arm_vectors
from armcopy.sim import Humanoid, ArmIK

S = 192
out = sys.argv[1]; specs = sys.argv[2:]
net = MoveNet(192); ik = ArmIK(Humanoid(64, 64))
imgs, Q, valid, cid, fid, clipped, names = [], [], [], [], [], [], []
summary = {}
for spec in specs:
    neg = spec.startswith("neg:"); name, folder = spec[4:].split("=", 1) if neg else spec.split("=", 1)
    files = sorted(glob.glob(folder + "/*.jpg")); step = 4 if neg else 1
    c = len(names); names.append(name); q_seq = []; nv = 0
    gap = True
    for i in range(0, len(files), step):
        img = cv2.imread(files[i]); h, w = img.shape[:2]; s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8); sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        imgs.append(cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        v = None if neg else arm_vectors(net(img), size=(w, h))
        if v is None:
            Q.append(np.zeros(3, np.float32)); valid.append(False); clipped.append(False); gap = True
        else:
            if gap: ik.q = ik.REST.copy()
            q, (eu, ef) = ik.fit(*v); gap = False
            ok = eu < 15 and ef < 15
            Q.append(q.astype(np.float32)); valid.append(bool(ok)); clipped.append(bool(ik.clipped)); nv += ok
        cid.append(c); fid.append(i)
    summary[name] = dict(frames=len(range(0, len(files), step)), valid=int(nv), negative_only=neg)
    print(name, summary[name], flush=True)
Q, valid, cid = np.array(Q), np.array(valid), np.array(cid)
# centred smoothing of labels over neighbouring valid frames (teacher jitter), per clip
for c in range(len(names)):
    idx = np.flatnonzero(cid == c)
    for k in idx:
        if valid[k]:
            nb = [j for j in range(max(idx[0], k - 2), min(idx[-1], k + 2) + 1) if valid[j]]
            Q[k] = Q[nb].mean(0)
np.savez_compressed(out, images=np.stack(imgs), q=Q, valid=valid, clip_id=cid, frame_id=np.array(fid), clipped=np.array(clipped), clip_names=np.array(names))
print(json.dumps(summary), "total", len(Q), "valid", int(valid.sum()))
