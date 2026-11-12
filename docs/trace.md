# The device traversal's error bounds

Why `src/vulkan/trace.comp` answers byte for byte what tinybvh answers on the CPU while
running most of its arithmetic in f32: the model, every filtered decision's bound,
and what is assumed. The shader's comments name these sections. The f32 data the
bounds rely on is built on the host: each node's image by `node_images` in
src/vulkan/vulkan.hpp, each triangle's by `triangle32` in src/vulkan/scene.hpp.

Checked, not assumed: tests/test_gpu.py compares the device with the CPU byte for
byte on random and adversarial rays, with and without instances, and
tests/test_golden.py holds both to one corpus of hashed answers. What it costs
against the CPU is `bench/rays.py`; the README has its table.

How the kernel is driven (src/vulkan/vulkan.hpp): rays go to the device in chunks of
a quarter million, two in flight, each chunk one submission. The host writes a
chunk's rays into mapped host memory, checking each value as it copies it, and
then submits; the shader reads them from there. Host memory is written only
before the submission that reads it. Each array of rays is laid out by one word,
the words from one ray's row to the next: 0 for one row for every ray, 3 for
float32 rows (the array was float32, or every value of it is exactly one, zero
or normal), 6 for float64 rows. The kernel runs each ray's whole crossing loop
(src/bvh.cpp's `crossings`, operation for operation), counts its crossings and
appends each one to the chunk's log on the device, in whatever order the lanes
run. src/vulkan/layout.comp scans the counts into offsets, gives each log entry
its slot in ray order, and writes the crossings into mapped host columns, a lane
per slot; the dispatches that do that are sized by the device from its own
count. The host reads the count back and widens the columns into the result.
When more crossings were counted than the log holds, nothing was laid out: the
host makes the log large enough and submits the chunk again.

## 1. Summary

Reference: `BVH<double, uint64_t>::IntersectOctant` / `IsOccludedOctant` with
`WATERTIGHT_TRITEST` (Woop's test), compiled without FMA contraction. Every decision that
traversal makes is a comparison of f64 values:

| decision | tinybvh | the kernel |
|---|---|---|
| child hit: `tmax >= tmin` | f64 slab test | f32 slab test + bound E (3.2); exact if the gap is within the bound |
| child order: `dist1 > dist2` | f64 | f32 + 2E; equality proofs for ties (3.4); exact otherwise |
| triangle rejected by sign of U, V, W | f64 | f32 + bound E_tri (4.3); exact if any barycentric is within its bound of 0 |
| det == 0 | f64 | implied by the sign certificate (5.1) |
| t >= 0 | f64 | sign of T against det in f32 + bound E_T (5.1); exact if T within bound of 0 |
| t < hit.t | f64 | interval comparison (5.5); exact if the intervals overlap |
| the reported t | f64 Woop | tinybvh's f64 Woop, computed once per hit after the traversal |

The kernel runs the same sequence of node visits and triangle tests as the CPU and reaches
the same outcome for each, so it reports the same primitive and the same bits of t. Nothing
relies on the boxes' padding or on a superset argument. The one structural difference from
tinybvh's own traversal is *when* the exact distance of an accepted triangle is computed:
not at acceptance (where the lanes of a warp disagree about whether they need it, and a
warp-level f64 instruction costs the same with one lane active as with thirty-two), but at
the end of the traversal, all lanes together. Until then the accepted triangle is held as a
certified interval around an accurate f32 estimate; a decision the interval cannot settle
resolves it on the spot.

## 2. The arithmetic model

### 2.1 What the CPU computes
IEEE-754 binary64, round to nearest even, unit roundoff u64 = 2^-53, no FMA
(`-ffp-contract=off`; tinybvh's `tinybvh_fma(double)` is `a * b + c`), denormals kept
(a denormal result has absolute error <= 2^-1075). Every f64 expression in `src/vulkan/trace.comp`'s
exact paths is the same expression in the same order marked `precise`, so the device computes
the same bits (verified on 10M rays before the filter was added).

### 2.2 What the device computes in f32 (verified on the 3090)
- `+`, `-`, `*`: correctly rounded to nearest, unit roundoff u = 2^-24. A product followed by
  an add may be fused into one FMA with one rounding; every bound below is derived for the
  unfused (two-rounding) case and the fused case has strictly less error.
- Flush to zero: a denormal *input* of `+ - *` is read as 0 and a denormal *result* becomes 0.
  Model: `fl(x op y) = (x' op y')(1 + d) + e` with `|d| <= u`, `|e| <= L = 2^-126`, and `x'`,
  `y'` equal to `x`, `y` unless denormal, when they are 0 (`|x' - x| <= L`). For a product the
  input flush contributes at most `L (|x| + |y|)`, for a sum at most `2L`. Every bound carries
  these as its "floor" terms.
- f64 -> f32 conversion: correctly rounded, denormals preserved (`|fl32(x) - x| <= u |x|`).
- Division: not correctly rounded (`7/3` came back 1 ulp off). Where a *value* depends on a
  quotient the kernel certifies it at run time by its residual (3.1, 5.3); where only a
  *bound* is divided, the Vulkan guarantee (OpFDiv within 2.5 ulp, i.e. relative error <= 5u)
  is assumed and covered by SLACK (2.3).
- `max`, `min` are exact selections. The order of evaluation is as written (GLSL does not
  reassociate without fast-math decorations, which glslang does not emit).

Assumed: f32 operands never overflow in the certified paths. The slab test is guarded per ray
(3.5); everywhere else an infinite or NaN intermediate makes the bound infinite or NaN, and
every certificate is written as a comparison that is false for NaN or infinity, so it falls
to the exact path (2.4).

### 2.3 Headroom conventions
Derivations below give first-order bounds with explicit constants. The shader evaluates them
with:
- `TWO_U = 2^-23 = 2u` in place of u: a factor 2 that absorbs the `(1 + u)^k` factors, the
  second-order terms where they are shown to be smaller than the first-order ones, and the
  rounding of the bound's own arithmetic (each bound is at most ~20 f32 operations on
  positive terms: relative error < 20u << 1).
- `FLOOR = 2^-120 = 64 L` in place of L: a factor 64 for the flush terms.
- `SLACK = 1 + 2^-16` on bounds that contain divisions: covers 2.5-ulp divisions (<= 5u each)
  and the roundings of the surrounding arithmetic (<= 20u) with 12x to spare.
- Comparison operands that are themselves rounded sums (`t + beta`, `best.t32 - hit_bound`)
  are moved outward by `round_up` / `round_down`: `x (1 +- 4u) +- FLOOR`, which bounds the
  exact sum from the right side. For the slab gap `tmax - tmin` and the tie tests the
  rounding of the difference (relative u) is covered because every operand's magnitude is
  itself at most E / (10u) (E contains 22u times those magnitudes), so the error of the
  difference is below 0.2 E, within the factor-2 headroom.
- Flushed terms inside a bound: each bound ends with an additive FLOOR-multiple at least
  2^-120, larger than any sum of flushed contributions (each < 2^-126, fewer than 64 of them).

### 2.4 Fail-safe comparisons
A certificate is always of the form "this f32 comparison is true", with NaN or infinity making
it false and sending the decision to the exact path: `abs(gap) > bound`, `U > E`, `U < -E`,
`abs(T) > E_T`, `det_lo > 0`, `Q_lo > 0`, `hi < +inf`, `hi < best_lo`, `lo >= best_hi` (the last
only after `hi` was shown finite, so `lo` is finite). A REJECT by `(T > 0) != all_positive` is
reached only after `abs(T) > E_T` succeeded, so T is finite and nonzero.

## 3. The node test

tinybvh, per child, with `rD = safercp(D)`, `ro = O * rD` (f64):
```
t1_k = fl(fl(b1_k * rD_k) - ro_k)      b1 = pos_k ? aabbMin_k : aabbMax_k     (entry slabs)
t2_k = fl(fl(b2_k * rD_k) - ro_k)      b2 = pos_k ? aabbMax_k : aabbMin_k     (exit slabs)
tmin = max(t1_x, t1_y, t1_z, 0)        tmax = min(t2_x, t2_y, t2_z, hit.t)
hit iff tmax >= tmin; dist = hit ? tmin : FAR; visit the child with the smaller dist first
(swap iff dist1 > dist2), push the other if hit.
```
`pos_k = rD_k >= 0`. Since `safercp` returns `1/D` or `copysign(2^1000, D)`, `pos_k` is the sign
bit of D_k being clear, for every D including -0.0 and NaN; the shader takes it from the sign
bit (`safe_reciprocal` clamps by the sign bit too, as copysign does: `x < 0 ?` would send
-0.0 to +2^1000 where tinybvh gives -2^1000).

### 3.1 The f32 reciprocal, certified by its residual
`D' = fl32(D)`, `rD' = D' divided in f32` (accuracy unknown), `r = fma(rD', D', -1)` with
`precise` (one rounding). If `|r| <= 4u` then `|rD' D' - 1| <= 4u (1 + u) < 4.01u` (fused), or
`<= 4.01u + u(1 + 4.01u) < 5.02u` if the implementation did not fuse. So `rD' = (1 + e) / D'`
with `|e| <= 5.1u`, and with `D' = D (1 + d)`, `|d| <= u`, and tinybvh's `rD = (1/D)(1 + d64)`:
```
|rD' - rD| <= 5.1u/|D'| + 1.01u/|D| + u64/|D| <= 6.3u |rD'|          (3.1a)
```
The shader uses 7u. A component `D_k = +-0` is not divided at all (3.6); a ray failing the
residual test on any other component runs every decision exact (`exact_only`).

### 3.2 The slab bound E
Per axis k let B_k = max(|root.lo'_k|, |root.hi'_k|) from the f32 root box. Rounding is
monotonic, so every node coordinate b satisfies `|b'| <= B_k` and `|b| <= B_k (1 + u)`.
Let w = max_k |O'_k| (`omega`), so `|O_k| <= 1.01 w`.

`ro' = fl(O' rD')`:
```
|O' rD' - O rD| <= u|O||rD'| + |O| 7u|rD'| = 8u |O||rD'|
rounding u|O' rD'|, flushes L(|O'| + |rD'| + 1)
|ro' - ro| <= 9.2u |ro'| + L(|O'| + |rD'| + 2)                        (3.2a)
```
(using `|O'||rD'| <= 1.01 |ro'| + 1.01 L`, since `ro'` is the rounded, possibly flushed product.)

Slab value, `t' = fl(fl(b' rD') - ro')` or fused, against `t = fl64(fl64(b rD) - ro)`:
```
|b' rD' - b rD| <= u|b||rD'| + |b| 7u|rD'| = 8u |b||rD'|
product rounding (unfused) <= u|b||rD'|(1 + u); subtraction rounding <= u(|b||rD'|(1 + u)^2 + |ro'|)
flushes: L(|b'| + |rD'|) + L (product) + 2L (subtraction inputs) + L (result)
f64 side: u64 (2|b||rD| + |ro|): 2^-29 of the f32 terms, absorbed
|t' - t| <= 10.1u B_k |rD'_k| + 10.3u |ro'_k| + L(B_k + 2|rD'_k| + w + 6)
```
The shader's `E_k = 2u * 11 (B_k |rD'_k| + |ro'_k|) + 2^-120 (B_k + 2|rD'_k| + w + 5)` is
at least twice the u-part and sixty times the L-part. `E = max_k E_k` is one constant per ray.

Then `|tmin' - tmin| <= E` and `|tmax' - tmax| <= max(E, hit_bound)`, because max and min are
1-Lipschitz in the sup norm (`|max_i x_i - max_i y_i| <= max_i |x_i - y_i|`), the clamp 0 is
exact in both, and hit.t is covered by 3.3.

### 3.3 hit.t in the f32 test
- No hit yet: tinybvh's `hit.t = 1e300`, the shader's `best.t32 = +inf` with `hit_bound = 0`.
  If every exit slab `t2'_k` and E are finite, `t2_k <= t2'_k + E < 2^129 < 1e300`, so
  `min(t2, 1e300) = min(t2)` and `min(t2', inf) = min(t2')`: the sentinel arm is inactive on
  both sides and the Lipschitz bound applies with 0 for it. If some `t2'_k` or E is not finite
  the gap is not finite and every certificate fails (2.4).
- Resolved hit: `t32 = fl32(t)`, `hit_bound = 2u |t32| + 2^-120` (conversion error u|t|,
  doubled; the floor guards a denormal t32 flushed inside min).
- Pending hit: `t32` and `hit_bound = beta` from section 5: `|t32 - t| <= beta`.

### 3.4 Decisions
gap' = tmax' - tmin'. With `G = E + max(E, hit_bound)`:
- `gap' > G` => `tmax - tmin >= gap' - G > 0` => hit certain.
- `gap' < -G` => miss certain.
- otherwise exact.
Both children certain and both hit: d' = tmin'_a - tmin'_b, `|tmin_a - tmin_b - d'| <= 2E`:
- `d' > 2E` => dist1 > dist2 => swap; `d' < -2E` => no swap.
- `|d'| <= 2E`: a tie within the bound. Two equality proofs (then dist1 == dist2, no swap):
  1. Both clamped to exactly 0: an axis k is a *candidate* for the f64 maximum of child a if
     `t1'_a,k + E >= tmin'_a - E`; 0 is a candidate if `0 >= tmin'_a - E`. The f64 maximum is
     attained by a candidate (a non-candidate j has `t1_a,j <= t1'_a,j + E < tmin'_a - E <=`
     the f64 value of the f32 maximiser). If 0 is the only candidate for both children, both
     f64 entry distances are exactly 0.
  2. Same single axis with equal coordinates: if axis k is the only candidate for both children
     and the parent's tie mask says the two children's f64 coordinate on that side of axis k is
     the same number, then `t1_a,k` and `t1_b,k` are the same f64 operations on the same inputs,
     hence equal, hence `tmin_a == tmin_b`. The mask has 6 bits per interior node (3 axes x
     min/max) computed on the host from the f64 nodes; the side is `pos_k ? min : max`.
  Otherwise exact. Measured: without the tie proofs 0.92% of node pairs go exact on the random
  rays, with them 0.18%; the sphere mesh's siblings share a coordinate 75% of the time, which
  is typical of any mesh (adjacent regions share boundary vertices).

One child certain hit and the other uncertain, or either uncertain: the whole pair goes exact
(the exact test recomputes rD and ro in f64 from the ray buffer and reads the f64 nodes, so no
f64 ray state is kept in registers for it).

### 3.5 Per-ray guards (`exact_only`)
All decisions of a ray are exact when any of: the residual test fails on a non-zero component
(NaN, or an inaccurate division); `B_k |rD'_k| + |ro'_k| >= 2^125` for some k (the slab
products could overflow, which the model excludes: then `|b' rD'| <= 2^125` and
`|t'| <= 2^126`); some non-zero `|D'_k| < 2^-100` (a denormal `D'_kx` flushed inside
`Sx' = D'_kx rD'_kz` would cost up to `L |rD'_kz|`, which with `|D'_kz| >= 2^-100` is below
2^-26 = 0.25u and is absorbed in 4.2; below that it is not); E not finite; every component
zero; or a zero component with `|O'_k| >= 2^24` (3.6).

### 3.6 A zero direction component
Axis-aligned rays are the common case of an orthographic view, so a component `D_k = +-0`
(by its f64 bits: a tiny non-zero one is 3.5's) is certified separately, without its slab
arithmetic. tinybvh computes that axis with `R = +-2^1000` (its clamped reciprocal, the sign
the zero's), `ro = fl(O_k R)` and `t = fl(b R - ro)` for the entry and exit planes b.

With `|O_k| < 2^24`, `O_k R` is exact and finite, so t is never NaN: `b R` is exact or an
infinity of b's sign, and the difference has the sign of `(b - O_k) R`, zero only for
`b = O_k`.

*The margin.* The shader has `O'_k = float(O_k)`, converted in a rounding mode Vulkan does
not fix, and `lo'_k`, `hi'_k` rounded to nearest on the host. Let `below(x)` and `above(x)`
move x by 8 ulps and 2^-120, as the shader computes them (two operations, each within an
ulp, and operands flushed by less than 2^-126). Then `lo'_k < below(O'_k)` puts a float
strictly between `lo'_k` and `O'_k`: so `lo_k` is under the midpoint above `lo'_k`, `O_k` is
over the float below `O'_k`, and `O_k - lo_k` exceeds half an ulp of a float, at least
2^-150. The same for `hi'_k > above(O'_k)`, and for the two outside comparisons.

*Certainly inside*, `lo'_k < below(O'_k)` and `hi'_k > above(O'_k)`: tinybvh's entry value
on the axis is at most `-2^850` and its exit value at least `2^850`, for either sign of the
zero (a negative zero swaps the planes and the sign of R together). Some component is not
zero, and its exit value is under 2^127 (3.5), so the axis decides neither
`tmin = max(.., 0)` nor `tmax = min(.., hit.t)`: both are those of the other axes, and E
bounds them as before. The shader gives the axis `-inf` and `+inf`, which decide neither.

*Certainly outside*, `lo'_k > above(O'_k)` or `hi'_k < below(O'_k)`: tinybvh's entry and exit
values on the axis are both at least `2^850` or both at most `-2^850`, so `tmin >= 2^850`
over a `tmax < 2^127`, or `tmax <= -2^850 < 0 <= tmin`: a miss, whatever the other axes say.

*Otherwise* the origin is within the margin of a plane, and the pair of children is tested
exactly.

The tie proofs of 3.4 never name a zero axis: its entry value `-inf` is never within E of
`tmin`. The triangle test needs nothing new: `D'_k = D_k = 0` exactly, a zero `D'_kx` makes
`Sx' = 0 = Sx` exactly (so 4.2's bound holds with room), and `kz` is the largest component,
never a zero one unless all three are (3.5).

## 4. The triangle test: rejection

tinybvh (f64), with `kx, ky, kz` from `maxdim(D)` (kx, ky swapped if D_kz < 0),
`Sz = rD_kz`, `Sx = fl(D_kx Sz)`, `Sy = fl(D_ky Sz)`:
```
A = fl(v1 - O), B = fl(v2 - O), C = fl(v0 - O)        (componentwise)
Ax = fl(A_kx - fl(Sx A_kz)), Ay = fl(A_ky - fl(Sy A_kz)), same for B, C
U = fl(fl(Cx By) - fl(Cy Bx)), V = fl(fl(Ax Cy) - fl(Ay Cx)), W = fl(fl(Bx Ay) - fl(By Ax))
reject unless (U >= 0 && V >= 0 && W >= 0) || (U <= 0 && V <= 0 && W <= 0)
det = fl(fl(U + V) + W); reject if det == 0
Az = fl(Sz A_kz), ...; T = fl(fl(fl(U Az) + fl(V Bz)) + fl(W Cz))
t = fl(T * fl(1 / det)); accept iff t >= 0 && t < hit.t
```
Rejection by signs is certain when one of U, V, W is certainly positive and another certainly
negative. (Zero counts for both signs in tinybvh, so "U within its bound of 0" is never a
rejection certificate; it is AMBIGUOUS and goes exact.)

### 4.1 Magnitudes
Per triangle the host stores `radius` >= max over its 9 f64 coordinates of |v|, rounded up.
Per ray `w = max |O'_k|`. `M = radius + w`: then `|v_k| + |O_k| <= 1.01 M` for the three
corners. The f32 triangle corners are `v' = fl32(v)` (host, RN): `|v' - v| <= u|v|`.

### 4.2 The sheared coordinates
`A'_k = fl(v'_k - O'_k)` against `A_k = fl64(v_k - O_k)`:
```
|A'_k - A_k| <= u|v_k| + u|O_k| + u|v'_k - O'_k| + 3L + u64|A_k| <= 2.02u M + 3L =: alpha
|A'_k| <= 1.02 M
```
`Sx' = fl(D'_kx rD'_kz)` against `Sx = fl64(D_kx fl64(1/D_kz))`: relative error of Sx' is
`(1 + 1 + 5.1 + 1)u` (two conversions, the certified reciprocal, one rounding) plus the flush
terms of 3.5; `|D_kx / D_kz| <= 1`, so `|Sx' - Sx| <= 8.5u + L` and `|Sx'| <= 1.01`. Same for Sy.
`Ax' = fl(A'_kx - fl(Sx' A'_kz))` (or fused) against `Ax = fl64(A_kx - fl64(Sx A_kz))`:
```
|Sx' A'_kz - Sx A_kz| <= |Sx' - Sx||A_kz| + |Sx'||A'_kz - A_kz| <= 8.6u M + 1.01 alpha + 1.01 L M
product rounding <= 1.03u M; input flush L(|Sx'| + |A'_kz|) <= L(1.01 + 1.02 M); result flush L
subtraction rounding <= u|Ax'| <= 2.05u M; flush L
|Ax' - Ax| <= alpha + 8.6uM + 1.01 alpha + 1.03uM + 2.05uM + L(2.03M + 3.1)
           <= 15.8u M + L(2.03 M + 9.2)
```
Define `a = 16u M + 8L(M + 1) >=` that. The same a bounds Ay', Bx', By', Cx', Cy'.

### 4.3 The barycentrics
`U' = fl(fl(Cx' By') - fl(Cy' Bx'))` or fused, against `U = fl64(fl64(Cx By) - fl64(Cy Bx))`:
```
|Cx' By' - Cx By| <= |Cx' - Cx||By'| + |Cx||By' - By| <= a(|Cx'| + |By'|) + a^2
product roundings u(|Cx' By'| + |Cy' Bx'|); subtraction rounding u|U'| <= u(|Cx' By'| + |Cy' Bx'|)(1 + u)
flushes L(|Cx'| + |By'| + |Cy'| + |Bx'| + 3)
```
Let `L1 = |Ax'| + |Ay'| + |Bx'| + |By'| + |Cx'| + |Cy'|` (the shader's `L`). Each 4-term sum of
magnitudes is <= L1, and `|Cx' By'| + |Cy' Bx'| <= (|Cx'| + |Cy'|)(|Bx'| + |By'|) <= L1^2 / 4`
(AM-GM on two numbers whose sum is <= L1). Then
```
|U' - U| <= a L1 + 2a^2 + (u/2) L1^2 (1 + u) + L(L1 + 3)
```
and identically for V and W (their product pairs are the other two pairings of the three 2D
vectors). Expanding `a L1 + 2a^2`:
```
a L1 = 16u M L1 + 8L(M + 1) L1
2a^2 = 512 u^2 M^2 + 256 u L M (M + 1) + 128 L^2 (M + 1)^2
```
The shader's `E = 2u (16 M L1 + L1^2 / 2) + 2^-38 M^2 + 2^-116 (M + 1)(M + L1 + 1)`:
- `2u * 16 M L1 = 32u M L1 >= 16u M L1 (1 + small)`; `2u L1^2 / 2 = u L1^2 >= (u/2) L1^2 (1 + u)`.
- `2^-38 M^2 = 1024 u^2 M^2 >= 512 u^2 M^2`.
- `2^-116 (M + 1)(M + L1 + 1)` covers `8L(M + 1)L1 + L(L1 + 3) <= 9L(M + 1)(L1 + 1)` (since
  `2^-116 = 1024 L`), `256uL M(M + 1) = 2^-142 M(M + 1)`, and `128 L^2 (M + 1)^2 = 2^-245 (M + 1)^2`.
If `16 M L1` or `L1^2` flush to zero in f32 the lost term was below 2^-126 and the 2^-116 term
is still present (its factors are >= 1). An infinite or NaN M or L1 makes E infinite or NaN.

### 4.4 Certificates
- REJECT: `(U' > E || V' > E || W' > E) && (U' < -E || V' < -E || W' < -E)`: some barycentric
  is > 0 in f64 and another < 0, tinybvh rejects.
- All three `> E` (or all `< -E`): all three have the same strict sign in f64; continue to 5.
- Otherwise AMBIGUOUS: tinybvh's own test runs (edge and vertex hits, degenerate triangles).

## 5. The triangle test: acceptance and the deferred distance

### 5.1 det, the sign of t
With U, V, W all > 0 in f64 (the mirror case is symmetric): `fl64(U + V) >= max(U, V) > 0`
(rounding is monotonic and sums of positives do not flush on the CPU), then `det > 0`. So
`det != 0` and `sign(det) = sign(U)`.

`t = fl64(T * fl64(1/det))`: `fl64(1/det)` is nonzero with det's sign (|det| < 2^1024 keeps 1/det
above the smallest denormal). `t >= 0` iff `T * fl64(1/det) >= 0` iff `sign(T) == sign(det)` or
`T == 0`, provided the product does not underflow to -0 (then `-0 >= 0` would accept). Below,
T is certified nonzero through `|T'| > E_T` on f32 values, so `|T| >= |T'| - E_T >= 2^-149`, and
`|det| <= 2^129`, so `|T/det| >= 2^-278 > 2^-1074`: no underflow, `sign(t) = sign(T) sign(det)`.

Bounds on the f32 shadows (`Sz' = rD'_kz`, `s = |Sz'|`):
```
Az' = fl(Sz' A'_kz) vs Az = fl64(Sz A_kz):
  |Sz' - Sz||A_kz| + |Sz'||A'_kz - A_kz| <= 7u s 1.01M + s alpha = 9.1u s M + 3 s L
  rounding <= 1.02u s M; flushes L(s + 1.02 M + 1)
  |Az' - Az| <= 10.2u s M + L(4s + 1.02M + 1)   ;  shader a_z = 2u * 6 s M + 2^-120 (3s + 2M + 1)
det' = fl(fl(U' + V') + W') vs det:
  |det' - det| <= 3E + u|U' + V'| + u|det'|(1 + u) + 5L <= 3E + 2.01u S3 + 5L,  S3 = |U'| + |V'| + |W'|
  shader E_det = 3E + 2 * 2u S3 + 2 * 2^-120
T' = fl(fl(fl(U' Az') + fl(V' Bz')) + fl(W' Cz')) vs T:
  |U' Az' - U Az| <= E|Az'| + |U'| a_z + E a_z   (and the two analogues)
  roundings <= 3.01u P,  P = |U' Az'| + |V' Bz'| + |W' Cz'|;  flushes L(S3 + Z + 7),  Z = |Az'| + |Bz'| + |Cz'|
  |T' - T| <= E Z + a_z S3 + 3 E a_z + 3.01u P + L(S3 + Z + 7)
  shader E_T = E Z + a_z S3 + 3 E a_z + 3 * 2u P + 2^-120 (S3 + Z + 5)
```
Certificates: `abs(T') > E_T` else AMBIGUOUS (a hit at the origin); then `(T' > 0) !=
all_positive` => `t < 0` => REJECT; `det_lo = |det'| - E_det > 0` else AMBIGUOUS (needed below).

### 5.2 An accurate f32 distance from the plane equation
Woop's f32 barycentrics are poor when the origin is far from a small triangle (A = v - O is a
long vector and U, V, W are tiny differences of its products: on the test scene their relative
error is ~1%), so `T'/det'` is a bad estimate of t for the culling comparisons. In exact
arithmetic t is the ray parameter at the triangle's plane, which Woop's `T/det` equals (the
sheared ray is `(0, 0, t)` and `T/det` is the barycentric interpolation of the sheared z's), and
which the plane equation also gives:
```
t_real = ((v0 - O) . n) / (D . n),   n = (v1 - v0) x (v2 - v0)
```
with no cancellation against the origin in `D . n` and a benign one in `(v0 - O) . n`.
The host stores `n' = fl32(n64)`, `n64` the f64 cross product of the f64 edges, and `nu`, a
bound on `|n'_k - n_k|` for every component k:
```
e1 = fl64(v1 - v0) = e1_real (1 + d), |d| <= u64; same for e2
n64_x = fl64(fl64(e1_y e2_z) - fl64(e1_z e2_y)):
  |n64_x - n_x| <= 3.01 u64 (|e1_y e2_z| + |e1_z e2_y|) + u64 |n64_x|
|n'_x - n_x| <= that + u |n64_x|
nu = max_k [4 u64 (|e1_i||e2_j| + |e1_j||e2_i|) + 2 u64 |n64_k| + u |n64_k|] * (1 + 2^-20), rounded up
```
(`triangle32` in src/vulkan/scene.hpp; the `(1 + 2^-20)` covers that arithmetic's own rounding.)

Shadows: `C' = fl(v0' - O')` (`|C'_k - C_k| <= alpha = 2.02uM + 3L`, where here C_k is the exact
real difference; `|C'_k| <= 1.02M`), `D' = fl32(D)` (`|D'_k - D_k| <= u|D_k|`, no flush in
conversion). With `C1 = sum |C'_k|`, `n1 = sum |n'_k|`, `D1 = sum |D'_k|`, `PN = sum |C'_k n'_k|`,
`PQ = sum |D'_k n'_k|`:
```
N' = fl(fl(fl(C'_x n'_x) + fl(C'_y n'_y)) + fl(C'_z n'_z)) vs N = (v0 - O) . n (exact):
  |C'_k n'_k - C_k n_k| <= alpha |n'_k| + (|C'_k| + alpha) nu
  roundings <= u PN + 2u PN (1 + 2u); flushes L(C1 + n1 + 5)
  |N' - N| <= (2.02uM + 3L) n1 + nu (C1 + 6.06uM + 9L) + 3.01u PN + L(C1 + n1 + 5)
  shader E_N = 2u (1.1 M n1 + 1.6 PN) + nu (C1 + 4 * 2u M + 1) + 2^-120 (C1 + 2 n1 + 5)
Q' = fl(fl(fl(D'_x n'_x) + fl(D'_y n'_y)) + fl(D'_z n'_z)) vs Q = D . n (exact):
  |D'_k n'_k - D_k n_k| <= u(1 + u)|D'_k||n'_k| + (1 + u)|D'_k| nu
  roundings <= 3.01u PQ; flushes L(D1 + n1 + 5)
  |Q' - Q| <= 4.02u PQ + 1.01 nu D1 + L(D1 + n1 + 5)
  shader E_Q = 2u * 2.1 PQ + nu (D1 + 2u D1) + 2^-120 (D1 + n1 + 5)
```
Checks: `2u * 1.1 M n1 = 2.2uM n1 >= 2.02uM n1`; `2u * 1.6 PN = 3.2u PN >= 3.01u PN`;
`nu * 4 * 2u M = 8uM nu >= 6.06uM nu`; `nu * 1 >= 9L nu`; `2^-120 (C1 + 2n1 + 5) >= L(C1 + n1 + 5) + 3L n1`;
`2u * 2.1 PQ = 4.2u PQ >= 4.02u PQ`; `nu(D1 + 2u D1) >= 1.01 nu D1`.

### 5.3 The quotient, certified by its residual
`t' = N'/Q'` in f32 (accuracy unknown), `rho = fma(t', Q', -N')` with `precise`:
`|t' Q' - N'| <= |rho| / (1 - u) <= |rho|(1 + 1.01u)`. Requiring `Q_lo = |Q'| - E_Q > 0`
(else AMBIGUOUS), `|Q| >= Q_lo`, `|N| <= |N'| + E_N`:
```
|t' - t_real| <= |t' - N'/Q'| + |N' - N|/|Q'| + |N| |Q' - Q| / (|Q'||Q|)
             <= [ |rho|(1 + 1.01u) + E_N + (|N'| + E_N) E_Q / Q_lo ] / |Q'|
shader beta_mt = that with (1 + 2 * 2u), times SLACK, plus 2^-120
```
SLACK covers the two 2.5-ulp divisions and the ~8 roundings of this expression (2.3).

### 5.4 tinybvh's own f64 distance against t_real
tinybvh's `t = fl64(T64 * fl64(1/det64))` where T64, det64 are its f64 Woop values. Its error
against the exact t_real has two parts.

(i) `|T64/det64 - t_real| <= (E_T64 + (|T64| + E_T64) E_det64 / |det64 - E_det64|) / |det64|`,
by the same algebra as 5.3, where E_T64, E_det64 bound the f64 computation's errors against
the exact quantities. The f64 computation has the same operation structure as the f32 shadow
with every error source smaller: inputs exact instead of rounded (Sx64 has relative error
2.01u64 against the 8.5u of Sx', Sz64 u64 against 7u, A64 u64|A| against alpha), unit
roundoff u64 = 2^-29 u, no flush (denormal results err by 2^-1075 < 2^-29 L). Hence
`E_T64 <= 2^-29 E_T*` and `E_det64 <= 2^-29 E_det*` where the starred bounds are the f32
formulas evaluated at the f64 magnitudes. Those magnitudes differ from the f32 ones by the
f32 errors (`S3_64 <= S3 + 3E`, `Z_64 <= Z + 3 a_z`, `P_64 <= P + E_T`, `|T64| <= |T'| + E_T`,
`|det64| >= det_lo`), so `E_det* <= E_det (1 + 4u)` and `E_T* <= E_T + 6 E a_z + 6u E_T <=
3.01 E_T` (using `3 E a_z <= E_T`). Therefore
```
|T64/det64 - t_real| <= 3.02 * 2^-29 * quotient_bound,   quotient_bound = (E_T + (|T'| + E_T) E_det / det_lo) / det_lo
```
and the shader uses `F64_SHARE = 4 * 2^-29` on `quotient_bound`.

(ii) The two final roundings: `|t - T64/det64| <= 2.01 u64 |T64/det64| <= 2^-52 1.01 (|t'| +
beta_mt + 4 * 2^-29 quotient_bound)`; the shader uses `F64_QUOTIENT = 2^-50` on that sum.

`beta = beta_mt + beta_64` then satisfies `|t' - t| <= beta` for tinybvh's t.

### 5.5 Accepting against the current best
tinybvh accepts iff `t < hit.t`. The current best is either resolved (`t32 = fl32(hit.t)`,
`hit_bound = 2u|t32| + 2^-120`), pending (`|t32 - hit.t| <= hit_bound = beta` of its own
certificate), or the sentinel (`t32 = +inf`, `hit_bound = 0`). With `hi = round_up(t' + beta)`,
`lo = round_down(t' - beta)`, `best_lo = round_down(t32 - hit_bound)`, `best_hi = round_up(t32 +
hit_bound)` (2.3):
- `hi` not finite: AMBIGUOUS (an overflowed quotient says nothing).
- `hi < best_lo`: `t <= hi < best_lo <= hit.t`: ACCEPT. The triangle becomes the pending best
  with `(t', beta)`; a pending previous best is simply dropped (tinybvh overwrote it too).
- `lo >= best_hi`: `t >= hit.t`: REJECT.
- otherwise AMBIGUOUS. This is where exact ties (`t == hit.t`, tinybvh keeps the earlier
  triangle) land: the exact test's strict `<` decides exactly as the CPU.

### 5.6 When the pending best is resolved
`resolve_best` runs tinybvh's f64 test on the pending triangle with limit FAR: the certificate
established that the sign tests pass and `t >= 0`, and the computed t does not depend on the
limit, so it returns the same t the CPU stored. It is called (a) before an AMBIGUOUS triangle's
exact test (which needs the exact hit.t as its limit), (b) before an exact node pair test
(which needs the exact hit.t in `min(.., hit.t)`), (c) once after the traversal. Case (c) is
where almost all of the f64 work happens, with the whole warp converged. A contradicted
certificate would show as a mismatch against the CPU; none has.

### 5.7 Occlusion
`IsOccluded` returns at the first accepted triangle; its distance is never reported. An ACCEPT
certificate therefore ends the query with no f64 at all; AMBIGUOUS runs the exact test (and
the sentinel is the only possible best, so nothing to resolve).

## 6. Why the answers are identical
By induction over the CPU's traversal: before each step the GPU holds the same current node,
the same stack, the same `hit.prim`, and knowledge of `hit.t` as either the exact value or a
certified interval containing it. At an interior node both kernels decide (hit1, hit2, swap);
each GPU decision is either certified (sections 3.4, with the interval for hit.t) or computed
by tinybvh's f64 operations after resolving hit.t, so it equals the CPU's; the stack and next
node then agree. At a leaf the triangles are tested in the same order; each test's outcome
(reject / accept) is certified or exact (4.4, 5.1, 5.5), and on accept the CPU's new hit.t is
the triangle's f64 t, which the GPU holds exactly or within the certified `beta`. The
traversals end at the same step. The reported prim is the last accepted triangle on both; the
reported t is tinybvh's f64 t of that triangle, computed by the same operations in `resolve_best`.
For occlusion both return true at the same first accepted triangle, or false together.

Tie cases and how they are handled: equal entry distances of siblings (3.4: proofs or exact);
a barycentric exactly 0 (edge or vertex hit: AMBIGUOUS, exact); `T` exactly 0 (origin on the
surface: AMBIGUOUS, exact); equal distances of two triangles (5.5: AMBIGUOUS, exact);
`t = -0.0` accepted by tinybvh (`-0 >= 0`): arises only for `T = 0` or an underflowing product,
both excluded from certificates (5.1), so exact.

## 7. What is assumed, what is not rigorous, what was not done
- Assumed of the device: correctly rounded f32 `+ - *` (Vulkan requires it); `precise fma()`
  is a single rounding (GLSL requires it; the two certificates that use it, 3.1 and 5.3, are
  derived to survive an unfused implementation as well); OpFDiv within 2.5 ulp for the handful
  of divisions inside *bounds* (3.2's E does not divide; 5.3 and 5.4 do; SLACK covers 256 ulp);
  f64 operations correctly rounded with `precise` honoured (the existing kernel's assumption,
  verified by the identity results); no reassociation of f32 expressions.
- Assumed of the CPU build: no FMA contraction, round to nearest, no flush to zero, binary64
  throughout (no x87 extended intermediates).
- Flush-to-zero is modelled for inputs and results of `+ - *` only; `min`/`max`/`abs`/
  comparisons are treated as exact selections. The NVIDIA FMNMX instruction may flush denormal
  inputs; a flushed denormal in `min(t2', hit32)` or `max(t1', 0)` would move the value by
  < 2^-126, which every bound's FLOOR term (>= 2^-120) exceeds.
- Not covered by the certificates (always exact, correct but at f64 speed): rays with a
  non-zero direction component of magnitude below 2^-100, with a zero one and an origin 2^24
  or more out on that axis, or whose slab products exceed 2^125; rays with |O| so large that the bound exceeds the triangles (around 1e3 to 1e4 scene
  extents on this mesh) run their triangle tests exact; rays through vertices and edges run
  the hit triangle(s) exact (that is the correct fallback, not a gap).
- Host-side bounds (`radius`, `nu`) are computed in f64 by `triangle32` and inflated by
  `(1 + 2^-20)` and one `nextafter`; this covers that arithmetic's own roundings.
- Indices are 32-bit on the device. `node_images` refuses a tree of 2^31 nodes or leaf
  slots, and arrays past what the device binds are refused at upload, so the CPU answers
  those scenes. A chunk is assumed to have fewer than 2^32 crossings (its count is u32).
- The top level (`trace_top`) carries the ray into each instance's frame in f64 exactly as
  tinybvh does and filters the instance's mesh like a lone one; an f32 instance transform
  with its own bounds was not done. Also not done: a BVH4 regrouping (it could not replicate
  the interior order); ray sorting and persistent threads (orthogonal to the filter).

