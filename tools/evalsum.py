import json,sys,glob
for p in sorted(glob.glob(sys.argv[1]+"*/script_free_evaluation.json")):
    s=json.load(open(p))["summary"]
    print(p.split("/")[-2], {k:round(s[k],2) for k in ("walking_success_rate","valid_left_step","valid_right_step","alternating_steps","fall_count","length") if k in s})
