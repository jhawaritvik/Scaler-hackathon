# Building an RL Environment for the Benefit of the Environment

When I first sat down to think about what RL environment to build, I got the usual suggestions everyone gets. Supply and demand simulators. Coding agents. Inventory management. Trading bots. The kind of stuff that shows up on every "RL project ideas" list.

None of it felt right. They were all the same well-trodden ground. Useful, sure, but I wanted to aim a bit more ambitiously. If I was going to spend weeks on this, I wanted the thing the agent learns to do to actually matter.

So I flipped the question. Instead of asking "what is a clean RL problem", I asked "what is a real problem worth solving where RL might help".

That gave me a tagline I kept coming back to. An RL environment for the benefit of the environment.

Once I had that framing, wildfires were the obvious pick. They are getting worse every year, the dispatch problem is genuinely hard, and the underlying physics is rich enough to be interesting without being so chaotic that learning is hopeless. A real problem with real impact.

The first version I asked Claude to build was quick. Maybe too quick. What came back was a grid where cells caught fire if their neighbors were on fire, with a wind variable you could nudge to change the spread probability. It worked. It even looked like fire if you squinted at the renders. But it was the cartoon version of a wildfire. The kind of thing where you draw flames as orange triangles.

The problem was not that it was wrong. The problem was that it was flat. There were no underlying dynamics for an agent to discover. No reason for the model to learn that this fire on this slope at this time of day was different from one a few cells over. Every fire was basically the same fire. That is not an environment, that is a screensaver.

The example that kept rattling around in my head was this loop. A forest catches fire. The air above it heats up. Hot air rises. That creates a low pressure pocket. Surrounding wind rushes in toward the fire. The fire gets more oxygen. It grows bigger. Which heats more air. And so on.

That is a firestorm. And the important part is that nobody coded it as a rule. It just falls out when you model the base variables honestly. That was the kind of texture I wanted in my environment. Things that were not scripted but that emerged on their own when the underlying physics was right.

Now, modelling every such phenomenon explicitly is impossible. Fire science is its own field with its own PhDs. So I made a call. Instead of trying to model the phenomena, I would model the base variables honestly and let the phenomena emerge.

I sat down with the codebase and started layering things in. Wind speed and direction, not just a scalar but a vector with its own daily cycle and random drift. Direction in the convention real fire models use, where it points to where the wind is coming from. Humidity tied inversely to the daily cycle, so things dry out during peak heat the way they actually do. Temperature with a base value plus a daily swing that peaks around 70 percent through the episode. Fuel moisture per cell, drifting toward equilibrium at fuel-class-specific time constants of 1 hour for grass and 10 hours for brush and forest. This is the Rothermel time-lag model from the 1972 paper.

I added elevation and slope, with heat transfer multiplied by an exponential of the slope angle, so cells uphill from a fire receive 2 to 3 times more heat than downhill neighbors. This is why uphill runs are so dangerous in real life. I added aspect, where north-facing slopes get different temperatures and different drying multipliers than south-facing ones, so south faces stay dry all day while north faces recover overnight. I added fuel types, grass and brush and forest, with different burn durations of 2, 4, and 6 ticks and different moisture extinction polynomials.

Then I asked AI to dig up the actual research on how these variables interact and wired the formulas in with real coefficients. The Alexandridis 2008 formulation gives wind-directional spread anisotropy, which is what creates the elliptical fire perimeters pointing downwind that you see in real fire maps. The Rothermel moisture damping polynomial is non-linear, so a small moisture change near the extinction point flips a fire from uncontrollable to stoppable. Spotting, where embers fly ahead of the main fire and start new ignitions, kicks in only when wind is above 28.8 km per hour and intensity is above 3.0.

The piece I am most proud of is the airflow potential field. It is a hidden scalar field that burning intensity feeds into. It diffuses and decays each step. The gradient of that field perturbs the local wind by up to plus or minus 4 km per hour. I never coded a firestorm rule. But because intense fires now generate their own pressure gradient that pulls wind toward them, the firestorm loop emerges from the physics. Big fires literally accelerate their own spread by inducing a draft. The exact phenomenon I had been imagining at the start, falling out of the model on its own.

That was the moment the environment stopped being a screensaver.

Once the fire side was honest, the firefighting side had to match. A high-fidelity fire model paired with a cartoon dispatch model would be useless. The agent would just learn to abuse the cheap abstractions. So I went back through NWCG doctrine and modelled six resource classes, each with realistic preparation times, travel speeds, and return-to-base overheads.

Hand crews are 20-person Type 2 ground crews with hand tools. They take 10 minutes to prep, 2.5 minutes per cell to travel, and 15 minutes to return. Wildland engines are Type 3 to 6 pumpers with foam, faster to deploy and faster to return. Helicopters carry water in a Bambi bucket of 300 to 700 gallons, take 5 minutes to prep and only 0.6 minutes per cell of travel, and they have smart routing that scoops from the nearest lake instead of base if the lake is closer. Air tankers are large fixed-wing planes carrying 2000 to 4000 gallons of retardant, with a long 30 minute return time so each drop is precious. Dozers are slow to roll out at 15 minutes prep but their lines are permanent firebreaks, and their production rate depends on the fuel type they are cutting through. Smokejumpers are the fastest to deploy at 2 minutes prep with a 0.3 minute jump, but they need an extraction helicopter to come pick them up, so they pay a 15 minute extraction overhead on the way back.

The point of all this granularity is not realism for its own sake. It is that the agent has to learn trade-offs that mirror real incident command decisions. Smokejumpers are fastest in but slowest out. Air tankers are devastating but each drop is expensive and slow to reload. Dozers are slow to start but their lines pay off if you commit early. Helicopters near water get free reloads, so positioning matters.

Each suppression action has a different mechanical effect too. Wet line saturates fuel moisture to 0.90 for a few steps until the moisture drifts back toward equilibrium. Retardant deposits a persistent ignition threshold bonus capped at plus 0.40 that decays at 0.008 per step normally and 0.010 per step per adjacent burning cell, calibrated against the USDA effectiveness window of 4 to 5 hours for Phos-Chek LC-95. That is the level of grounding I wanted. Every constant in the file should be defensible against a published source.

The reward function took more iterations than the physics. The physics has the courtesy of being grounded in equations. The reward function is just you, alone, trying to anticipate every way your agent will cheat you. And it will cheat you, every single time.

The first version was naive. Plus one for every cell saved, minus one for every cell burned. Sounds reasonable. The agent immediately learned it could do nothing on small fires that would burn out on their own and still rack up positive reward from the unburned cells around them. Passive reward farming.

So I added an active fire pressure penalty of minus 0.005 per burning cell. Now the agent had urgency. But on the hard 25 by 25 grid that uniform negative signal drowned out everything else. The agent learned that the optimal policy was to give up early and minimize regret. So I added a grid-size normalization factor that scales penalties down on bigger maps, so the positive learning signal does not get buried.

Then I caught the agent doing the reverse. It would dispatch a single hand crew to nibble at the edge of a fire that was about to self-extinguish anyway and claim credit for containment. So I gated the containment bonus. It only fires if the agent actually suppressed something on that tick. You do not get credit for the fire going out unless you helped put it out.

Here is where I landed.

For the per-step environmental signals, a burning structure gives minus 0.12 times its priority each step it stays on fire, as ongoing urgency. A structure lost gives a one-time minus 0.50 times priority when it transitions from burning to fully burned, tracked in a set so it cannot fire twice. A structure that is safe and under threat gives a small plus 0.003 times priority each step. Cells suppressed give plus 0.06 each, a strong immediate signal for transitioning a cell from burning to suppressed. Containment gives plus 0.018 per net burning cell reduced, gated on the agent having suppressed something. Cells protected give plus 0.0025 each, capped at 12 cells so it does not get trivialized. Active fire pressure gives minus 0.005 per burning cell, capped at 8 cells so it does not dominate the gradient. Fire fully extinguished gives a one-time terminal bonus that scales from plus 0.30 if it took a long time to plus 0.70 if it was instant.

Then there are the action costs, applied separately. Direct attack costs minus 0.0022. Line construction costs minus 0.0037. Wet line costs minus 0.0030. Water drop costs minus 0.0060. Retardant drop costs minus 0.0090, the most expensive of the suppression types because air tankers should be reserved for the cases that need them. Backfire costs minus 0.0112, the highest, because it is a powerful but dangerous tool. There is a per-resource dispatch surcharge between minus 0.0004 and minus 0.0030 depending on the platform. Invalid actions cost minus 0.05. And if units are sitting available with no work, an idle penalty of minus 0.005 per available unit kicks in, capped at minus 0.02.

The principles I converged on were these.

Reward transitions, not states. A burning structure is bad, but the moment it becomes fully burned is when the big penalty fires. This makes the gradient cleaner. The agent learns what action prevented the bad outcome, not just that some state is bad.

Weather is information, not penalty. I never penalize the agent for high winds. Weather is something it observes and conditions on, not something it gets blamed for. Penalizing weather would teach the agent learned helplessness.

One-time bonuses for one-time events. Fire extinguished, structure lost, these fire exactly once via tracking sets. No double counting.

Cap penalties to keep the gradient signal alive. The fire pressure penalty is capped at 8 cells. The protected bonus at 12. Without caps, on a hard map, one term dominates and the others become invisible.

Gate positive signals on agent contribution. The containment bonus only triggers if the agent suppressed something this tick. No credit for nature taking its course.

Building this was less about reinforcement learning and more about honest modelling of a complex system. The RL part is almost incidental. Give a competent algorithm an honest environment and it will find policies. Give it a dishonest one and it will find exploits.

The biggest shift in how I think now is this. Do not try to script the interesting behavior. Model the substrate honestly and let the interesting behavior emerge. I never wrote a firestorm rule. I wrote a buoyancy field with realistic decay and diffusion constants, and firestorms appeared. I never wrote a rule that fires spread faster uphill. I wrote a slope coefficient and an elevation grid, and uphill runs emerged.

That is the bar I want to hold future environments to. If the only way an interesting phenomenon shows up is because I hard-coded it, the model is not deep enough yet. Find the underlying variables. Wire them up honestly. Let the emergence do the work.

And if the agent ends up learning to position fires near lakes for free helicopter reloads, that is not a bug. That is the environment doing its job.

A note on where this fits in the hackathon themes. The environment lands most naturally under Theme 3.1, World Modeling for Professional Tasks. The theme description talks about environments that require real interaction with dynamic systems, where the model is expected to do real hard work instead of exploiting shortcuts, and where the agent has to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows. That maps almost word for word to what this environment does. The reward function is explicitly designed to prevent shortcut exploitation. The agent has to track fuel moisture drift, fleet status across the prep and travel and operate and return cycle, and forecasted weather changes. The causal reasoning piece is exactly the firestorm story, the agent has to learn that wind and slope and fuel and buoyancy compose causally, not as independent features. And wildfire incident command is a professional task in the most literal sense, there are NWCG-certified incident commanders whose entire job is this dispatch problem.

There is also a strong secondary fit to Theme 2, Long-Horizon Planning. Rewards are sparse and delayed, the fire-extinguished terminal bonus only fires once at the end of the episode. Multi-step reasoning is forced by the resource model, dozers take 15 minutes of prep plus travel time before they cut their first foot of line, so commitments must be made many steps before the payoff arrives. And the recover-from-early-mistakes property holds, if you misallocate the air tanker early it is gone for 30 minutes and you have to adapt the rest of your plan around that gap.

Built for the benefit of the environment, one Rothermel coefficient at a time.
