# Where does UMA-AIF belong?

Assessed against the two existing repositories.

## kaysentinel — no fit

[kaysentinel](https://github.com/kaymyg/kaysentinel) is a Rust protocol-integrity
runtime for blockchain state execution: it extracts canonical state mutations
from Geth/Reth traces into a Structural Sufficient Representation, commits them
with domain-separated BLAKE3, and checks cross-client consensus at the mutation
level.

Different language, different domain, different threat model. The only shared
vocabulary is the word "consensus", and it means something else in each — cross-
client agreement on state transitions there, agreement between reference model
subspaces here. Nothing to merge.

## ames — adjacent, but deliberately closed to this class of signal

[ames](https://github.com/kaymyg/ames) is the near miss, and it is worth being
precise about why it is still a miss.

AMES is a nine-layer containment framework: L1–L4 prevent an agent from leaving
its execution environment, L5–L7 detect, L8 commits evidence to a hash-linked
BLAKE3 ledger, and L9 terminates. It is Python, it is about AI agent safety, and
L6 is literally called "Cognitive Drift". On the surface UMA-AIF looks like it
wants to be an L6 implementation.

Three properties of AMES's own design say otherwise.

**1. AMES's evidence/testimony split puts UMA-AIF on the wrong side of the
line.** The L8 ledger records host-observed evidence only — VM lifecycle events,
image measurements, broker decisions. Guest-reported claims are quarantined into
a reserved actor-ID band and excluded from evaluation. UMA-AIF reads activation
tensors from inside the model the agent is running. In AMES's taxonomy that is
testimony, not evidence: it is produced by the thing under observation, and a
model that could manipulate its own activations to evade the auditor is exactly
the adversary the framework assumes. Feeding UMA-AIF output into L8 as evidence
would break the invariant the ledger exists to maintain.

**2. L6 is advisory by construction.** AMES structurally prevents cognitive
drift from triggering a kill — there is no parameter on the L9 path that accepts
a drift finding. That is a deliberate choice, and a good one: drift is a soft
signal and L9 is an irreversible action. A statistical activation auditor is a
softer signal still. Merging UMA-AIF into L6 would add code that, by design,
cannot affect any outcome.

**3. The deployment surfaces do not overlap.** AMES contains an agent process
inside an ephemeral microVM and watches it from the host. It does not need — and
mostly cannot have — access to model weights or forward-pass internals, because
the agents it contains typically call a model over an API that terminates
outside the VM. UMA-AIF requires hooks into the transformer itself. Different
side of the trust boundary, and a torch dependency that a containment runtime
targeting Rust for production should not inherit.

AMES also has no plugin or detector interface. Its architecture is closed on
purpose, in service of "structural impossibility over monitoring". UMA-AIF is
monitoring, and probabilistic monitoring at that.

## Recommendation: a separate repository, cross-referenced

The two are complements, and the complementarity is worth stating explicitly in
both READMEs because it is the interesting part:

|  | AMES | UMA-AIF |
|---|---|---|
| Vantage point | Host, outside the guest | Inside the forward pass |
| Object of study | The agent process | The model's internal representation |
| Evidence class | Host-observed | Guest-internal (testimony) |
| Guarantee style | Structural impossibility | Statistical anomaly detection |
| Failure mode | Availability (a kill that shouldn't fire) | Calibration (a score that means nothing) |
| Enforcement | L9 terminates | Advisory; projection at most |

AMES answers "did the agent leave the box". UMA-AIF asks "did the computation
inside the box drift from its references". Neither subsumes the other, and each
is weakened by pretending to be the other.

## The one integration worth building

If a link is wanted later, the clean seam is **UMA-AIF as an evidence source for
an L8-style ledger, entered in the testimony band, never in the evidence band.**

Concretely: emit a per-step audit record — seed, sampled layer indices, raw
metrics, calibrated scores, calibrator fingerprint — and hash-chain it with the
same domain-separated BLAKE3 scheme. That gives a tamper-evident, replayable
audit trail with no claim to being host-observed and no path to L9.

Everything that record needs is already deterministic: `LayerSampler` and
`EnvelopeNoiseInjector` are seeded, `ScoreCalibrator.state_dict()` serialises the
baseline, and `test_audit_step_is_reproducible_from_seed` asserts a step can be
re-derived bit-for-bit from its inputs. That was designed in for this reason.

This is not implemented. It is a note about where the seam is, so that if the
two projects are ever joined it happens at the boundary that respects both
architectures rather than by dropping a torch dependency into a containment
runtime.
