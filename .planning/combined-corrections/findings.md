# Findings

- Two approved training recordings: human_1792298109792073701 (805 frames), human_1792302165387076969 (950 frames).
- Both final ten frames pass On, physical open fingers, open command and retreat gates; raw checksums must pass again before export.
- Together with public training: 39 episodes / 6155 frames. Human share under uniform frame sampling = 1755/6155, about 28.51%.
- Public holdout remains 10 episodes / 1170 frames. No new source downloads required.
- Existing exporter hardcodes 16 public episodes; extend with explicitly validated public-base manifest, retain legacy behavior.
- Existing curation review is pending visual review, not expert certification. User explicitly requests retaining full approved trajectories; preserve whole recordings and disclose limitation.
