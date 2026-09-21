# Audit: data/crops_plus_sdk/head_cap (the deployed head_cap trunk's data), 2026-09-17

`python scripts/113_audit_headcap_crops.py [--score]`. 1867 unique heads
(augmented copies collapsed): SDK apd_v3 val 1063, site snapshots 654, native
video frames 150. Five parallel Claude reviewers, one written rule set
(apd/schema.py HEAD_CAP_RULE). verdicts.csv has a note per head.

Rule decision taken during the audit (user, 2026-09-17): a dark navy beret-style
kitchen cap and a white chef toque are PRESENT. HEAD_CAP_RULE updated.

| label | heads | ok | wrong class | not a head | unclear |
|---|---|---|---|---|---|
| cap | 1023 | 989 | 11 | 2 | 21 |
| no_cap | 839 | 771 | 3 | 4 | 61 |
| in both folders | 5 | 4 resolved to no_cap | - | - | 1 |
| all | 1867 | 1760 (94%) | 14 (0.7%) | 6 | 83 (4%) |

By source, wrong class: site 12, native 2, SDK 0.

Wrong class: street caps labelled cap #80 #81 #171 #384 #498 #502 #1787; bare
heads labelled cap #495 #519 #553 #621; hairnets labelled no_cap #168 #258 #1735.
Not a head: #602 #672 #844 #1109 #1645 #1678.
Both folders (trained with both labels): #151 #532 #560 #1802 are no_cap; #161 unclear.

Second decision (user, 2026-09-17): cloth chef skullcaps, plain black or
striped, are PRESENT. 16 black-skullcap heads labelled no_cap are relabelled to
cap: #809 #810 #820 #909 #911 #912 #930 #931 #932 #934 #935 #989 #1255 #1256
#1258 #1589. Striped ones (#881 #883 ...) were already cap. After this:
ok 1759, wrong class 30 (1.6%), not a head 6, unclear 68.
Picture of every change: headcap_changes.jpg.
