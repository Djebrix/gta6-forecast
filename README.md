[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Djebrix/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Djebrix/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), from hour 1's print and the
hourly curves of 38+ past big game launches.

`main.ts` (Node 22, stdlib only) reads a case file, and for each of hours
2–24 writes five quantiles of that hour's viewer count. The forecast is a
launch-day shape model: per hour, the log-ratio of viewers to hour 1's print
is regressed on log hour-1 across the comparables, fitted twice (all launches
and console-only launches, which unlock at midnight local like GTA VI) and
blended by the case's hour-1 size, with a diurnal correction for the
US-unlock morning rise, the midday lull and the evening peak, and a spread
from the regressions' residual deviations.

Run it: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts`

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on make.fun.
