# Chapter 4/5 framing: what the evidence can support

Use the branch that matches the result; the choice is made by the decision rules in PREREGISTRATION.md, not by taste.

## If the quantum arm shows no advantage (or equivalence)
**Claim.** On one-step EUR/USD forecasting, simulated quantum feature maps and quantum-generator features did not
improve on inputs-only or matched classical controls; where equivalence was tested, differences were within the
pre-registered margins. This is a properly tested null, not an absence of evidence.

**Contributions that stand regardless of outcome.**
1. A diagnosis that the original QGAN-LLM forecast was independent of the generator by construction, and a redesign
   that places the generator on the inference path.
2. A matched-capacity, matched-input-width paired design, with persistence as the baseline and a residual target.
3. Synthetic-data fidelity of the QGAN versus classical GANs (H1), with the capacity confound bracketed.
4. Implementation feasibility and cost of simulating the circuits, with the architectural limits documented.

**Wording to avoid.** "No quantum advantage exists." Use: "no advantage was detected under this design, for these
circuits, on simulated hardware; equivalence within X was/was not shown."

## If the quantum arm beats both controls
**Claim.** A fixed PQC feature map improved one-step forecasts beyond persistence and beyond a matched classical
random feature map, on this period and design. Report the effect size and the seed-to-seed spread, state that it is
simulated and period-conditional, and do not extend it to hardware, other pairs, or other horizons.

## If results are mixed or inconclusive
Report each arm's effect with its CI, say which comparisons were underpowered, and give the sample size that would
be needed. Do not promote any single favourable metric.

## Fixed disclosure sentences
- "All quantum components were classically simulated; no runtime or hardware advantage is claimed."
- "Both a high-capacity and a parameter-matched classical generator were evaluated; results are reported against both."
- "The forecasting study uses one chronological test period; uncertainty across seeds reflects feature-map randomness only."
