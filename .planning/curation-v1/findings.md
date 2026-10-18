# Findings

Previous full mixture was an incorrect interpretation of user's request: source preservation does not mean all source actions belong in training. Existing39/6155 export remains historical, not the new recommendation.

Source805: closes0..549, opens550..804; source950: closes0..429, opens430..489, closes490..844, opens845..949. Last10 gates pass for complete sources, but that alone does not label intermediate actions expert quality.

Select continuous ranges per original source and keep each as a separate episode, so10-step action windows never cross a removed gap. Do not decimate frames, alter actions, synthesize success or claim new manual approval.

Candidate805[425,635):210frames; candidate950[770,906):136frames. Combined346, excluded1409fromthiscandidate (not deleted). Both start already holding bowl, so placement/release/retreat only. Last10allOn/opencommands/fingers>=.06m/eefclear>=.10m, bowlstable. Physicalcontrol.05s, video nominal10fps.

User asks to see candidate clips before deciding. Selection remains candidate; no actual curated dataset export and no training until feedback. FirefoxSnap cannot access hidden local paths, so read-only review server bound127.0.0.1:8769 serves only `.local/curation-placement-v1`; browser title verified correct andHTTP200. Raw originals remain inaccessible from that server root.
